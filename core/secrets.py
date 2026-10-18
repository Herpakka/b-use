"""Credentials the agent types without ever seeing them: ``{{SECRET:name}}`` placeholders.

The model only handles the placeholder. The browser layer swaps in the value at the moment it
fills a field, refuses fields outside the secret's domains (like a password manager), and
puts the placeholder back wherever the value shows up again in what it reads from the page.
Values are encrypted at rest (AES-GCM, key from ``BU_SECRETS_KEY``) and never leave the server.

Stores are scoped: the API uses one scope per tenant; the CLI and evals use ``BU_LOCAL_SECRETS``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol
from urllib.parse import quote, quote_plus

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, SecretStr

from core.config import StaticSecret
from core.domains import domain_allowed, normalize_domains
from core.logging import get_logger
from core.store import utcnow

log = get_logger("core.secrets")

SECRET_NAME = r"^[A-Za-z0-9_.-]{1,64}$"
PLACEHOLDER = re.compile(r"\{\{\s*SECRET:([A-Za-z0-9_.-]{1,64})\s*\}\}")
# Shorter values can't be redacted from page text without mangling it.
MIN_VALUE_LENGTH = 4


def placeholder(name: str) -> str:
    return "{{SECRET:" + name + "}}"


def secret_names(text: str) -> list[str]:
    """Names referenced in ``text``, in order of first appearance."""
    return list(dict.fromkeys(PLACEHOLDER.findall(text)))


@dataclass(frozen=True)
class Secret:
    name: str
    value: str = field(repr=False)
    # Hosts it may be typed on (subdomains included); empty = any domain the run allows.
    domains: tuple[str, ...] = ()


class SecretInfo(BaseModel):
    """What may be shown about a secret: never its value."""

    name: str
    domains: list[str]
    updated_at: datetime


def fill_placeholders(
    text: str, secrets: Mapping[str, Secret], host: str | None
) -> tuple[str, list[Secret]] | str:
    """Swap placeholders for values, or explain why not: unknown secret, or ``host`` is not
    one the secret is bound to. Returns ``(text, secrets used)`` or the refusal reason."""
    used: list[Secret] = []
    for name in secret_names(text):
        secret = secrets.get(name)
        if secret is None:
            return f"unknown secret {name!r}"
        if secret.domains and not (host and domain_allowed(host, secret.domains)):
            return (
                f"secret {name!r} may only be typed on {', '.join(secret.domains)},"
                f" not on {host or 'this page'}"
            )
        used.append(secret)
    return PLACEHOLDER.sub(lambda m: secrets[m.group(1)].value, text), used


def _mask(text: str, value: str, mask: str, min_match: int) -> str:
    """Replace ``value`` — or a prefix of it at least ``min_match`` long, as left by clipping
    — with ``mask``."""
    head = value[:min_match]
    out: list[str] = []
    i = 0
    while (j := text.find(head, i)) >= 0:
        k = len(head)
        while k < len(value) and j + k < len(text) and text[j + k] == value[k]:
            k += 1
        out += [text[i:j], mask]
        i = j + k
    out.append(text[i:])
    return "".join(out)


class SecretRedactor:
    """Puts the placeholder back wherever a typed value reappears: input values, page text,
    URLs (a GET form puts fields in the query string)."""

    # The DOM extractor clips long strings (to 40 chars at the shortest, ending in "…"), so a
    # long value may come back cut short; shorter values must match whole, so ordinary words
    # that happen to start a secret are left alone.
    MIN_PREFIX = 39

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def add(self, secrets: Iterable[Secret]) -> None:
        for s in secrets:
            for form in (s.value, " ".join(s.value.split()), quote(s.value), quote_plus(s.value)):
                if form:
                    self._values[form] = placeholder(s.name)

    def __bool__(self) -> bool:
        return bool(self._values)

    def scrub(self, text: str) -> str:
        for value in sorted(self._values, key=len, reverse=True):
            if value[: self.MIN_PREFIX] in text:
                text = _mask(text, value, self._values[value], min(len(value), self.MIN_PREFIX))
        return text


SecretResolver = Callable[[str, str], Awaitable[Secret | None]]
"""(scope, name) → the secret, or None if there is no such secret."""


class SecretsUnavailableError(RuntimeError):
    pass


class SecretStore(Protocol):
    async def setup(self) -> None: ...
    async def get(self, scope: str, name: str) -> Secret | None: ...
    async def put(self, scope: str, name: str, value: str, domains: Sequence[str]) -> None: ...
    async def delete(self, scope: str, name: str) -> bool: ...
    async def list(self, scope: str) -> list[SecretInfo]: ...


def _check(name: str, value: str) -> None:
    if not re.match(SECRET_NAME, name):
        raise ValueError(f"invalid secret name {name!r}")
    if len(value) < MIN_VALUE_LENGTH:
        raise ValueError(f"secret values must be at least {MIN_VALUE_LENGTH} characters")


def generate_key() -> str:
    """A fresh value for ``BU_SECRETS_KEY``."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode()


class SecretCipher:
    """AES-256-GCM. The associated data binds each ciphertext to its scope and name, so a row
    copied onto another tenant's or secret's row no longer decrypts."""

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("BU_SECRETS_KEY must be 32 bytes, base64-encoded")
        self._aead = AESGCM(key)

    @classmethod
    def from_setting(cls, key: SecretStr | None) -> SecretCipher | None:
        if key is None or not key.get_secret_value():
            return None
        try:
            raw = base64.urlsafe_b64decode(key.get_secret_value())
        except (binascii.Error, ValueError) as err:
            raise ValueError("BU_SECRETS_KEY is not valid base64") from err
        return cls(raw)

    def encrypt(self, plaintext: str, aad: str) -> bytes:
        nonce = os.urandom(12)
        return nonce + self._aead.encrypt(nonce, plaintext.encode(), aad.encode())

    def decrypt(self, blob: bytes, aad: str) -> str:
        return self._aead.decrypt(blob[:12], blob[12:], aad.encode()).decode()


class MemorySecretStore:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], tuple[Secret, datetime]] = {}

    async def setup(self) -> None:
        return None

    async def get(self, scope: str, name: str) -> Secret | None:
        item = self._items.get((scope, name))
        return item[0] if item else None

    async def put(self, scope: str, name: str, value: str, domains: Sequence[str]) -> None:
        _check(name, value)
        self._items[(scope, name)] = (Secret(name, value, normalize_domains(domains)), utcnow())

    async def delete(self, scope: str, name: str) -> bool:
        return self._items.pop((scope, name), None) is not None

    async def list(self, scope: str) -> list[SecretInfo]:
        return [
            SecretInfo(name=s.name, domains=list(s.domains), updated_at=at)
            for (sc, _), (s, at) in sorted(self._items.items())
            if sc == scope
        ]


class StaticSecretStore:
    """Read-only secrets from settings (``BU_LOCAL_SECRETS``), for the CLI and evals: one user,
    so the scope is ignored."""

    def __init__(self, secrets: Mapping[str, StaticSecret]) -> None:
        self._secrets: dict[str, Secret] = {}
        for name, s in secrets.items():
            _check(name, s.value.get_secret_value())
            self._secrets[name] = Secret(
                name, s.value.get_secret_value(), normalize_domains(s.domains)
            )

    async def get(self, scope: str, name: str) -> Secret | None:
        return self._secrets.get(name)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS bu_secrets (
    scope      text NOT NULL,
    name       text NOT NULL,
    ciphertext bytea NOT NULL,
    domains    jsonb NOT NULL DEFAULT '[]',
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (scope, name)
)
"""


class PostgresSecretStore:
    """Encrypted rows in ``bu_secrets``. Without a key (``BU_SECRETS_KEY`` unset) nothing can be
    stored or read, and the API says so instead of keeping plaintext."""

    def __init__(self, pool: AsyncConnectionPool[Any], cipher: SecretCipher | None) -> None:
        self.pool = pool
        self.cipher = cipher
        self._warned = False

    @property
    def available(self) -> bool:
        return self.cipher is not None

    def _require(self) -> SecretCipher:
        if self.cipher is None:
            raise SecretsUnavailableError(
                "secrets need BU_SECRETS_KEY (generate one with"
                ' `python -c "from core.secrets import generate_key; print(generate_key())"`)'
            )
        return self.cipher

    async def setup(self) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(_SCHEMA)

    async def get(self, scope: str, name: str) -> Secret | None:
        cipher = self._require()
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT ciphertext, domains FROM bu_secrets WHERE scope = %s AND name = %s",
                [scope, name],
            )
            row = await cur.fetchone()
        if row is None:
            return None
        try:
            value = await asyncio.to_thread(
                cipher.decrypt, bytes(row["ciphertext"]), f"{scope}/{name}"
            )
        except InvalidTag:
            # Wrong key (rotated?) or a tampered row: treat as missing, loudly.
            log.error("secret_undecryptable", scope=scope, name=name)
            return None
        return Secret(name, value, tuple(row["domains"]))

    async def put(self, scope: str, name: str, value: str, domains: Sequence[str]) -> None:
        _check(name, value)
        blob = self._require().encrypt(value, f"{scope}/{name}")
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO bu_secrets (scope, name, ciphertext, domains) VALUES (%s, %s, %s, %s)
                ON CONFLICT (scope, name) DO UPDATE
                SET ciphertext = EXCLUDED.ciphertext, domains = EXCLUDED.domains, updated_at = now()
                """,
                [scope, name, blob, Jsonb(list(normalize_domains(domains)))],
            )

    async def delete(self, scope: str, name: str) -> bool:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM bu_secrets WHERE scope = %s AND name = %s", [scope, name]
            )
            return bool(cur.rowcount > 0)

    async def list(self, scope: str) -> list[SecretInfo]:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "SELECT name, domains, updated_at FROM bu_secrets WHERE scope = %s ORDER BY name",
                [scope],
            )
            rows = await cur.fetchall()
        return [SecretInfo.model_validate(row) for row in rows]

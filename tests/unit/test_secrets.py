import pytest
from cryptography.exceptions import InvalidTag
from pydantic import SecretStr

from core.config import StaticSecret
from core.secrets import (
    MemorySecretStore,
    Secret,
    SecretCipher,
    SecretRedactor,
    StaticSecretStore,
    fill_placeholders,
    generate_key,
    secret_names,
)

PW = Secret("pw", "correct-horse-battery", ("shop.test",))


def test_placeholders_are_found_in_order_without_duplicates() -> None:
    text = "log in as {{SECRET:user}} / {{ SECRET:pw }} then {{SECRET:user}} again"
    assert secret_names(text) == ["user", "pw"]
    assert secret_names("{{SECRET:}} {SECRET:x} {{secret:x}}") == []


def test_fill_checks_existence_and_domain_binding() -> None:
    secrets = {"pw": PW, "free": Secret("free", "anywhere-value")}
    assert fill_placeholders("{{SECRET:pw}}", secrets, "login.shop.test") == (
        "correct-horse-battery",
        [PW],
    )
    assert fill_placeholders("x{{SECRET:free}}y", secrets, "other.test") == (
        "xanywhere-valuey",
        [secrets["free"]],
    )
    assert fill_placeholders("{{SECRET:nope}}", secrets, "shop.test") == "unknown secret 'nope'"
    refused = fill_placeholders("{{SECRET:pw}}", secrets, "evil-shop.test")
    assert refused == "secret 'pw' may only be typed on shop.test, not on evil-shop.test"
    assert isinstance(fill_placeholders("{{SECRET:pw}}", secrets, None), str)


def test_redactor_masks_exact_encoded_and_clipped_values() -> None:
    r = SecretRedactor()
    assert not r
    long_value = "tok_" + "a1b2c3d4" * 10  # 84 chars: the DOM extractor clips it
    r.add([Secret("email", "alice@example.com"), Secret("token", long_value)])
    assert r.scrub("user=alice@example.com") == "user={{SECRET:email}}"
    assert r.scrub("/login?user=alice%40example.com&x=1") == "/login?user={{SECRET:email}}&x=1"
    clipped = long_value[:59] + "…"
    assert r.scrub(f'value="{clipped}"') == 'value="{{SECRET:token}}…"'
    # A short value must match whole: words that merely start like it are left alone.
    assert r.scrub("alice@example.org") == "alice@example.org"


def test_cipher_round_trip_is_bound_to_scope_and_name() -> None:
    cipher = SecretCipher.from_setting(SecretStr(generate_key()))
    assert cipher is not None
    blob = cipher.encrypt("hunter22", "acme/pw")
    assert b"hunter22" not in blob
    assert cipher.decrypt(blob, "acme/pw") == "hunter22"
    with pytest.raises(InvalidTag):
        cipher.decrypt(blob, "globex/pw")  # copied onto another tenant's row
    assert SecretCipher.from_setting(None) is None
    with pytest.raises(ValueError, match="32 bytes"):
        SecretCipher.from_setting(SecretStr("c2hvcnQ="))


async def test_memory_store_scopes_and_validates() -> None:
    store = MemorySecretStore()
    await store.put("acme", "pw", "hunter22", ["Shop.Test", "*.shop.test"])
    assert await store.get("acme", "pw") == Secret("pw", "hunter22", ("shop.test",))
    assert await store.get("globex", "pw") is None
    assert [s.name for s in await store.list("acme")] == ["pw"]
    assert "hunter22" not in repr(await store.list("acme"))
    with pytest.raises(ValueError, match="at least 4"):
        await store.put("acme", "short", "abc", [])
    with pytest.raises(ValueError, match="invalid secret name"):
        await store.put("acme", "../x", "long enough", [])
    assert await store.delete("acme", "pw") and not await store.delete("acme", "pw")


async def test_static_store_ignores_scope() -> None:
    store = StaticSecretStore({"pw": StaticSecret(value=SecretStr("hunter22"), domains=["a.test"])})
    assert await store.get("anything", "pw") == Secret("pw", "hunter22", ("a.test",))
    assert await store.get("", "other") is None
    assert "hunter22" not in repr(await store.get("", "pw"))

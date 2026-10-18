from tests.agent_fakes import FakeBrowser
from tests.api_support import client_for, memory_factory


def test_healthz_returns_ok() -> None:
    with client_for(memory_factory(FakeBrowser(), [])) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

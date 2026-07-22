from typing import cast
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.service import RunService


def test_health_api__returns_ok_without_touching_run_service() -> None:
    service_mock = Mock(spec=RunService)
    service = cast(RunService, service_mock)

    with TestClient(create_app(service)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert service_mock.mock_calls == []

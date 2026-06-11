"""Tests for FastAPI server endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient

from server.deps import reset_state
from server.main import app


@pytest.fixture(autouse=True)
def _reset():
    """Reset app state between tests."""
    reset_state()


@pytest.fixture
def client():
    """FastAPI TestClient — synchronous, no real HTTP needed."""
    return TestClient(app)


class TestHealth:
    def test_health_endpoint(self, client: TestClient):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestConfig:
    def test_get_config(self, client: TestClient):
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert data["chat_model"] == "qwen2.5:7b"
        assert data["top_k"] == 3

    def test_update_config(self, client: TestClient):
        response = client.post("/api/config", json={"top_k": 10})
        assert response.status_code == 200
        data = response.json()
        assert data["top_k"] == 10

        # Reset
        client.post("/api/config", json={"top_k": 3})


class TestChat:
    def test_chat_validation(self, client: TestClient):
        """Missing question should get validation error."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422


class TestDocuments:
    def test_list_documents_empty(self, client: TestClient):
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["total_chunks"] == 0
        assert data["documents"] == []

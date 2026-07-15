import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.web.app import WebSettings, create_app


class FakeVerifier:
    def __init__(self, score=0.9):
        self.score = score
        self.calls = []

    def verify_against_references(self, capture, references, strategy):
        self.calls.append((capture, list(references), strategy))
        return SimpleNamespace(
            aggregated_score=self.score,
            match=self.score >= 0.3128704727,
        )


class WebRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        settings = WebSettings(
            database_path=root / "demo.sqlite3",
            reference_dir=root / "references",
            capture_dir=root / "captures",
            config_path=Path("config/model_config.json"),
        )
        self.verifier = FakeVerifier()
        self.app = create_app(settings=settings, verifier_provider=lambda: self.verifier)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_main_pages_and_health_are_available(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/users").status_code, 200)
        self.assertEqual(self.client.get("/history").status_code, 200)
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(
            health.json(),
            {
                "status": "OK",
                "model_config_found": True,
                "database_available": True,
                "model_name": "baseline_formal/baseline_con_aumento",
                "threshold": 0.3128704727,
            },
        )

    def test_registers_user_with_reference(self):
        response = self.client.post(
            "/users",
            data={"full_name": "Usuario Web", "rfid_uid": "WEB-001"},
            files=[("references", ("front.jpg", b"fake-image", "image/jpeg"))],
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Usuario Web", response.text)
        self.assertIn("WEB-001", response.text)

    def test_duplicate_uid_is_rejected_from_registration_route(self):
        files = [("references", ("front.jpg", b"fake-image", "image/jpeg"))]
        first = self.client.post(
            "/users",
            data={"full_name": "Usuario Uno", "rfid_uid": "DUP-001"},
            files=files,
            follow_redirects=True,
        )
        second = self.client.post(
            "/users",
            data={"full_name": "Usuario Dos", "rfid_uid": "dup-001"},
            files=files,
            follow_redirects=True,
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertIn("Ya existe un usuario con ese UID RFID", second.text)
        self.assertEqual(len(self.app.state.database.list_users()), 1)

    def test_unknown_uid_returns_denied_and_creates_history_entry(self):
        response = self.client.post(
            "/verify",
            data={"rfid_uid": "NOT-KNOWN"},
            files={"capture": ("capture.jpg", b"fake-image", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("DENIED", response.text)
        self.assertIn("RFID_UNKNOWN", response.text)
        self.assertIn("RFID_UNKNOWN", self.client.get("/history").text)
        self.assertEqual(self.verifier.calls, [])

    def test_smoke_flow_uses_mocked_inference_and_records_history(self):
        registration = self.client.post(
            "/users",
            data={"full_name": "Usuario Demo 1", "rfid_uid": "01020304"},
            files=[("references", ("front.jpg", b"fake-image", "image/jpeg"))],
            follow_redirects=True,
        )
        self.assertIn("Usuario Demo 1", registration.text)

        granted = self.client.post(
            "/verify",
            data={"rfid_uid": "01020304"},
            files={"capture": ("positive.jpg", b"fake-image", "image/jpeg")},
        )
        self.assertIn("GRANTED", granted.text)
        self.assertIn("RFID_AND_FACE_OK", granted.text)
        self.assertEqual(len(self.verifier.calls), 1)

        self.verifier.score = 0.1
        denied = self.client.post(
            "/verify",
            data={"rfid_uid": "01020304"},
            files={"capture": ("negative.jpg", b"fake-image", "image/jpeg")},
        )
        self.assertIn("DENIED", denied.text)
        self.assertIn("FACE_NO_MATCH", denied.text)

        history = self.client.get("/history")
        self.assertIn("GRANTED", history.text)
        self.assertIn("FACE_NO_MATCH", history.text)
        self.assertEqual(len(self.app.state.database.list_events()), 2)

    def test_inactive_user_is_denied_without_calling_inference(self):
        self.client.post(
            "/users",
            data={"full_name": "Usuario Inactivo", "rfid_uid": "OFF-001"},
            files=[("references", ("front.jpg", b"fake-image", "image/jpeg"))],
        )
        user = self.app.state.database.get_user_by_rfid("OFF-001")
        self.client.post(f"/users/{user.id}/toggle")

        response = self.client.post(
            "/verify",
            data={"rfid_uid": "OFF-001"},
            files={"capture": ("capture.jpg", b"fake-image", "image/jpeg")},
        )

        self.assertIn("DENIED", response.text)
        self.assertIn("USER_INACTIVE", response.text)
        self.assertEqual(self.verifier.calls, [])
        self.assertIn("USER_INACTIVE", self.client.get("/history").text)


if __name__ == "__main__":
    unittest.main()

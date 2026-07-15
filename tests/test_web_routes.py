import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.web.app import WebSettings, create_app


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
        self.client_context = TestClient(create_app(settings=settings))
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
        self.assertEqual(health.json()["threshold"], 0.3128704727)

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


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.web.database import Database
from src.web.services import DemoService
from src.web.storage import ImageStorage


class FakeVerifier:
    def __init__(self, score):
        self.score = score
        self.calls = []

    def verify_against_references(self, capture, references, strategy):
        self.calls.append((capture, list(references), strategy))
        return SimpleNamespace(
            aggregated_score=self.score,
            match=self.score >= 0.3128704727,
        )


class WebAccessFlowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.database = Database(root / "demo.sqlite3")
        self.database.initialize()
        self.storage = ImageStorage(root / "references", root / "captures")
        self.storage.initialize()

    def tearDown(self):
        self.temporary.cleanup()

    def service(self, score=0.8):
        verifier = FakeVerifier(score)
        service = DemoService(
            self.database,
            self.storage,
            "config/model_config.json",
            verifier_provider=lambda: verifier,
        )
        return service, verifier

    def add_user_with_reference(self, uid="KNOWN-01"):
        user = self.database.create_user("Usuario Demo", uid)
        self.database.add_reference(user.id, "reference-1.jpg")
        return user

    def test_unknown_user_is_denied_without_inference_and_event_is_recorded(self):
        service, verifier = self.service()

        result = service.verify_access("UNKNOWN-01", "capture.jpg")

        self.assertEqual(result.access_decision, "DENIED")
        self.assertEqual(result.reason, "RFID_UNKNOWN")
        self.assertEqual(verifier.calls, [])
        self.assertEqual(self.database.list_events()[0].reason, "RFID_UNKNOWN")

    def test_inactive_user_is_denied_without_inference(self):
        user = self.add_user_with_reference()
        self.database.set_user_active(user.id, False)
        service, verifier = self.service()

        result = service.verify_access("KNOWN-01", "capture.jpg")

        self.assertEqual(result.access_decision, "DENIED")
        self.assertEqual(result.reason, "USER_INACTIVE")
        self.assertEqual(verifier.calls, [])

    def test_known_rfid_and_matching_face_grants_access_using_max(self):
        self.add_user_with_reference()
        service, verifier = self.service(score=0.9)

        result = service.verify_access("KNOWN-01", "capture.jpg")

        self.assertEqual(result.access_decision, "GRANTED")
        self.assertEqual(result.reason, "RFID_AND_FACE_OK")
        self.assertEqual(result.threshold, 0.3128704727)
        self.assertEqual(verifier.calls[0][2], "max")
        self.assertEqual(self.database.list_events()[0].access_decision, "GRANTED")

    def test_known_rfid_and_non_matching_face_denies_access(self):
        self.add_user_with_reference()
        service, _ = self.service(score=0.1)

        result = service.verify_access("KNOWN-01", "capture.jpg")

        self.assertEqual(result.access_decision, "DENIED")
        self.assertEqual(result.reason, "FACE_NO_MATCH")
        self.assertFalse(result.face_match)


if __name__ == "__main__":
    unittest.main()

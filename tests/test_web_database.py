import tempfile
import unittest
from pathlib import Path

from src.web.database import Database


class WebDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temporary.name) / "demo.sqlite3")
        self.database.initialize()

    def tearDown(self):
        self.temporary.cleanup()

    def test_creates_user_and_reference(self):
        user = self.database.create_user("Ana Torres", "04:A1:B2")
        self.database.add_reference(user.id, "reference.jpg")

        stored = self.database.get_user_by_rfid("04:a1:b2")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.full_name, "Ana Torres")
        self.assertEqual(stored.reference_count, 1)
        self.assertTrue(stored.is_active)

    def test_rfid_uid_is_unique_case_insensitive(self):
        self.database.create_user("Ana Torres", "ABCD-1234")

        with self.assertRaisesRegex(ValueError, "Ya existe"):
            self.database.create_user("Otra persona", "abcd-1234")


if __name__ == "__main__":
    unittest.main()

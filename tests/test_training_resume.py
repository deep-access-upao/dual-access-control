import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.training.train import create_callbacks, read_history, resume_state


class TrainingResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.history_path = self.root / "history.csv"
        with self.history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "loss", "val_loss"])
            writer.writeheader()
            writer.writerows(
                [
                    {"epoch": 0, "loss": 0.8, "val_loss": 0.6},
                    {"epoch": 1, "loss": 0.4, "val_loss": 0.3},
                ]
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_history_keeps_completed_epochs(self):
        rows = read_history(self.history_path)
        self.assertEqual([row["epoch"] for row in rows], ["0", "1"])

    def test_resume_restores_next_epoch_and_global_best(self):
        model_path = self.root / "checkpoint.keras"
        model_path.touch()
        sentinel_model = object()
        with mock.patch(
            "src.training.train.tf.keras.models.load_model",
            return_value=sentinel_model,
        ) as load_model:
            model, initial_epoch, best_val_loss = resume_state(
                model_path, self.history_path
            )
        self.assertIs(model, sentinel_model)
        self.assertEqual(initial_epoch, 2)
        self.assertEqual(best_val_loss, 0.3)
        self.assertTrue(load_model.call_args.kwargs["compile"])

    def test_resumed_checkpoint_preserves_previous_best(self):
        callbacks = create_callbacks(
            self.root / "checkpoint.keras",
            self.history_path,
            patience=2,
            append_history=True,
            initial_best_val_loss=0.3,
        )
        self.assertEqual(callbacks[0].best, 0.3)
        self.assertTrue(callbacks[2].append)


if __name__ == "__main__":
    unittest.main()

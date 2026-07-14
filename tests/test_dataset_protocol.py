import csv
import unittest
from pathlib import Path

from src.dataset.audit_splits import audit
from src.dataset.build_manifest import MANIFEST_COLUMNS
from src.dataset.build_pairs import CSV_COLUMNS, output_path
from src.dataset.build_splits import SPLITS, assign_splits


def manifest_row(image, person, video, digest, split=""):
    return {
        "image_path": image,
        "person_id": person,
        "source_video": video,
        "source_mapping": "test",
        "view": "frontal",
        "status": "usable",
        "quality_status": "ok",
        "quality_reasons": "",
        "width": "112",
        "height": "112",
        "channels": "3",
        "file_size_bytes": "1",
        "sha256": digest,
        "split": split,
    }


class DatasetProtocolTests(unittest.TestCase):
    def test_split_is_reproducible_and_keeps_videos_whole(self):
        rows = []
        for person in ("p1", "p2"):
            for index in range(3):
                video = f"{person}/video_{index}"
                for frame in range(2):
                    rows.append(manifest_row(
                        f"{video}/frame_{frame}.jpg", person, video, f"{person}-{index}-{frame}"
                    ))

        first = assign_splits(rows, seed=42, ratios=(0.5, 0.25, 0.25))
        second = assign_splits(rows, seed=42, ratios=(0.5, 0.25, 0.25))
        self.assertEqual(first, second)

        video_splits = {}
        for row in first:
            video_splits.setdefault(row["source_video"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in video_splits.values()))
        for person in ("p1", "p2"):
            self.assertEqual({row["split"] for row in first if row["person_id"] == person}, set(SPLITS))

    def test_audit_fails_when_a_video_crosses_splits(self):
        root = Path("tests/.protocol_fixture")
        root.mkdir()
        try:
            manifest_path = root / "manifest.csv"
            pairs_dir = root / "pairs"
            pairs_dir.mkdir()

            rows = []
            for split in SPLITS:
                shared_video = "shared-video" if split in {"train", "validation"} else "test-video-a"
                rows.extend([
                    manifest_row(f"{split}/a.jpg", "p1", shared_video, f"{split}-a", split),
                    manifest_row(f"{split}/b.jpg", "p2", f"{split}-video-b", f"{split}-b", split),
                ])
            with manifest_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            for split in SPLITS:
                pair = {
                    "image_a": f"{split}/a.jpg",
                    "image_b": f"{split}/b.jpg",
                    "label": 0,
                    "person_a": "p1",
                    "person_b": "p2",
                    "source_video_a": "unused",
                    "source_video_b": "unused",
                    "view_a": "frontal",
                    "view_b": "frontal",
                    "split": split,
                }
                path = output_path(pairs_dir, split)
                with path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
                    writer.writeheader()
                    writer.writerow(pair)

            report, errors = audit(manifest_path, pairs_dir)
            self.assertEqual(report["shared_videos"], 1)
            self.assertTrue(any("videos compartidos" in error for error in errors))
        finally:
            for path in pairs_dir.glob("*.csv") if pairs_dir.exists() else []:
                path.unlink()
            if pairs_dir.exists():
                pairs_dir.rmdir()
            if manifest_path.exists():
                manifest_path.unlink()
            if root.exists():
                root.rmdir()


if __name__ == "__main__":
    unittest.main()

"""Persistencia SQLite para usuarios, referencias y eventos de acceso."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src.web.models import AccessEvent, FaceReference, User


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    rfid_uid TEXT NOT NULL UNIQUE COLLATE NOCASE,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    image_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS access_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rfid_uid TEXT NOT NULL,
    user_id INTEGER,
    score REAL,
    threshold REAL NOT NULL,
    face_match INTEGER NOT NULL CHECK (face_match IN (0, 1)),
    access_decision TEXT NOT NULL CHECK (access_decision IN ('GRANTED', 'DENIED')),
    reason TEXT NOT NULL,
    references_used INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_users_rfid_uid ON users(rfid_uid);
CREATE INDEX IF NOT EXISTS idx_access_events_created_at ON access_events(created_at DESC);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            connection.executescript(SCHEMA)
            connection.commit()

    def create_user(self, full_name: str, rfid_uid: str) -> User:
        created_at = utc_now()
        try:
            with closing(self.connect()) as connection:
                cursor = connection.execute(
                    "INSERT INTO users(full_name, rfid_uid, created_at) VALUES (?, ?, ?)",
                    (full_name, rfid_uid, created_at),
                )
                user_id = int(cursor.lastrowid)
                connection.commit()
        except sqlite3.IntegrityError as error:
            if "users.rfid_uid" in str(error):
                raise ValueError("Ya existe un usuario con ese UID RFID.") from error
            raise
        return User(user_id, full_name, rfid_uid, True, created_at)

    def delete_user(self, user_id: int) -> None:
        with closing(self.connect()) as connection:
            connection.execute("DELETE FROM users WHERE id = ?", (user_id,))
            connection.commit()

    def add_reference(self, user_id: int, image_path: str) -> FaceReference:
        created_at = utc_now()
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "INSERT INTO face_references(user_id, image_path, created_at) VALUES (?, ?, ?)",
                (user_id, image_path, created_at),
            )
            reference_id = int(cursor.lastrowid)
            connection.commit()
        return FaceReference(reference_id, user_id, image_path, created_at)

    def get_user_by_rfid(self, rfid_uid: str) -> User | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT u.*, COUNT(fr.id) AS reference_count
                FROM users u
                LEFT JOIN face_references fr ON fr.user_id = u.id
                WHERE u.rfid_uid = ? COLLATE NOCASE
                GROUP BY u.id
                """,
                (rfid_uid,),
            ).fetchone()
        return self._user(row) if row else None

    def get_user(self, user_id: int) -> User | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                """
                SELECT u.*, COUNT(fr.id) AS reference_count
                FROM users u
                LEFT JOIN face_references fr ON fr.user_id = u.id
                WHERE u.id = ?
                GROUP BY u.id
                """,
                (user_id,),
            ).fetchone()
        return self._user(row) if row else None

    def list_users(self) -> list[User]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT u.*, COUNT(fr.id) AS reference_count
                FROM users u
                LEFT JOIN face_references fr ON fr.user_id = u.id
                GROUP BY u.id
                ORDER BY u.created_at DESC, u.id DESC
                """
            ).fetchall()
        return [self._user(row) for row in rows]

    def set_user_active(self, user_id: int, is_active: bool) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                "UPDATE users SET is_active = ? WHERE id = ?", (int(is_active), user_id)
            )
            connection.commit()

    def list_references(self, user_id: int) -> list[FaceReference]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM face_references WHERE user_id = ? ORDER BY id",
                (user_id,),
            ).fetchall()
        return [
            FaceReference(row["id"], row["user_id"], row["image_path"], row["created_at"])
            for row in rows
        ]

    def record_event(
        self,
        *,
        rfid_uid: str,
        user_id: int | None,
        score: float | None,
        threshold: float,
        face_match: bool,
        access_decision: str,
        reason: str,
        references_used: int,
    ) -> int:
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO access_events(
                    rfid_uid, user_id, score, threshold, face_match,
                    access_decision, reason, references_used, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rfid_uid,
                    user_id,
                    score,
                    threshold,
                    int(face_match),
                    access_decision,
                    reason,
                    references_used,
                    utc_now(),
                ),
            )
            event_id = int(cursor.lastrowid)
            connection.commit()
            return event_id

    def list_events(self, limit: int = 200) -> list[AccessEvent]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT e.*, u.full_name AS user_name
                FROM access_events e
                LEFT JOIN users u ON u.id = e.user_id
                ORDER BY e.created_at DESC, e.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            AccessEvent(
                id=row["id"],
                rfid_uid=row["rfid_uid"],
                user_id=row["user_id"],
                user_name=row["user_name"],
                score=row["score"],
                threshold=row["threshold"],
                face_match=bool(row["face_match"]),
                access_decision=row["access_decision"],
                reason=row["reason"],
                references_used=row["references_used"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _user(row: sqlite3.Row) -> User:
        return User(
            id=row["id"],
            full_name=row["full_name"],
            rfid_uid=row["rfid_uid"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            reference_count=row["reference_count"],
        )

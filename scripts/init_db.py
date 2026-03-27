"""Initialize the InvoiceAgent database with all required tables."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.database import DatabaseManager


def main() -> None:
    db = DatabaseManager()
    db.init_db()
    print(f"Database initialized at {db._db_path}")  # noqa: T201 — script output


if __name__ == "__main__":
    main()

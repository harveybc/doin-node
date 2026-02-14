"""Migrate chain data from JSON file to SQLite database.

Usage:
    python -m doin_node.storage.migrate --json chain.json --db chain.db
    
Or programmatically:
    from doin_node.storage.migrate import migrate_json_to_sqlite
    migrate_json_to_sqlite("data/chain.json", "data/chain.db")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from doin_core.models.block import Block

from doin_node.storage.chaindb import ChainDB

logger = logging.getLogger(__name__)


def migrate_json_to_sqlite(
    json_path: str | Path,
    db_path: str | Path,
    batch_size: int = 100,
) -> int:
    """Migrate a JSON chain file to a SQLite database.

    Args:
        json_path: Path to the source chain.json file.
        db_path: Path for the new SQLite database.
        batch_size: Number of blocks to commit per batch (for progress).

    Returns:
        Number of blocks migrated.

    Raises:
        FileNotFoundError: If json_path doesn't exist.
        ValueError: If the database already has blocks.
    """
    json_path = Path(json_path)
    db_path = Path(db_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON chain file not found: {json_path}")

    if db_path.exists():
        # Check if it already has data
        db = ChainDB(db_path)
        db.open()
        if db.height > 0:
            existing = db.height
            db.close()
            raise ValueError(
                f"Database already contains {existing} blocks. "
                "Delete it first or use a different path."
            )
        db.close()

    # Load JSON chain
    logger.info("Loading JSON chain from %s...", json_path)
    raw = json.loads(json_path.read_text())
    blocks = [Block.model_validate(b) for b in raw]
    logger.info("Loaded %d blocks from JSON", len(blocks))

    if not blocks:
        logger.warning("No blocks to migrate")
        return 0

    # Write to SQLite
    db = ChainDB(db_path)
    db.open()

    migrated = 0
    for i, block in enumerate(blocks):
        try:
            db.append_block(block)
            migrated += 1

            if (i + 1) % batch_size == 0:
                logger.info("Migrated %d / %d blocks...", migrated, len(blocks))
        except Exception as e:
            logger.error("Failed at block #%d: %s", block.header.index, e)
            break

    db.close()

    logger.info(
        "Migration complete: %d / %d blocks migrated to %s",
        migrated, len(blocks), db_path,
    )

    # Show size comparison
    json_size = json_path.stat().st_size
    db_size = db_path.stat().st_size
    logger.info(
        "Size: JSON=%dKB → SQLite=%dKB (%.1f%%)",
        json_size // 1024,
        db_size // 1024,
        (db_size / json_size * 100) if json_size > 0 else 0,
    )

    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate DOIN chain data from JSON to SQLite",
    )
    parser.add_argument(
        "--json", required=True,
        help="Path to source chain.json file",
    )
    parser.add_argument(
        "--db", required=True,
        help="Path for destination chain.db file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Blocks per progress report (default: 100)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        count = migrate_json_to_sqlite(args.json, args.db, args.batch_size)
        print(f"\n✅ Successfully migrated {count} blocks")
    except Exception as e:
        print(f"\n❌ Migration failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

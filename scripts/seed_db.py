
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from loguru import logger

from app.config import settings
from app.middleware.auth import hash_password


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATABASE_URL = settings.database_url

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = PROJECT_ROOT / "seed" / "migrations"
DOCS_DIR = PROJECT_ROOT / "seed" / "docs"
MANIFEST_PATH = PROJECT_ROOT / "seed" / "ingestion_manifest.json"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".html",
    ".htm",
    ".txt",
    ".md",
}

INGESTION_VERSION = 2

DEMO_USERS = [
    ("agent@demo.local", "agent123", False),
    ("admin@demo.local", "admin123", True),
]


# --------------------------------------------------
# Document discovery
# --------------------------------------------------

def _collect_files(subdir: str) -> list[Path]:
    root = DOCS_DIR / subdir

    if not root.exists():
        return []

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and path.name != ".gitkeep"
    )


def _select_corpus():
    """Ingest only trusted Kubernetes documentation."""
    return _collect_files("true_data")


# --------------------------------------------------
# Document fingerprinting and manifest
# --------------------------------------------------

def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {
            "version": INGESTION_VERSION,
            "documents": {},
        }

    try:
        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )

        if not isinstance(manifest, dict):
            raise ValueError("Manifest must contain a JSON object")

        if not isinstance(manifest.get("documents"), dict):
            raise ValueError("Manifest documents field must be an object")

        return manifest

    except (json.JSONDecodeError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot read ingestion manifest: {MANIFEST_PATH}"
        ) from exc


def _save_manifest(manifest: dict) -> None:
    """
    Save the manifest atomically, retrying Windows/OneDrive
    file-lock errors before giving up.
    """

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Unique temporary file avoids collisions between runs.
    temp_path = MANIFEST_PATH.with_name(
        f"{MANIFEST_PATH.stem}.{os.getpid()}.tmp"
    )

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
            file.flush()
            os.fsync(file.fileno())

        for attempt in range(1, 11):
            try:
                os.replace(temp_path, MANIFEST_PATH)

                logger.debug("Ingestion manifest saved")
                return

            except PermissionError:
                if attempt == 10:
                    raise

                wait_seconds = min(attempt, 3)

                logger.warning(
                    "Manifest is locked by Windows/OneDrive. "
                    "Retrying in {} seconds ({}/10)",
                    wait_seconds,
                    attempt,
                )

                time.sleep(wait_seconds)

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning(
                    "Could not remove temporary manifest: {}",
                    temp_path,
                )


# --------------------------------------------------
# Embedding validation
# --------------------------------------------------

def _embed_document(
    embed_fn,
    texts: list[str],
) -> list[list[float]]:
    """
    Delegate batching, retries, and caching to embedding_service.py.
    """

    if not texts:
        return []

    embeddings = embed_fn(texts)

    if len(embeddings) != len(texts):
        raise ValueError(
            f"Embedding count mismatch: expected {len(texts)}, "
            f"got {len(embeddings)}"
        )

    for index, vector in enumerate(embeddings):
        if len(vector) != 1536:
            raise ValueError(
                f"Invalid embedding dimension at index {index}: "
                f"expected 1536, got {len(vector)}"
            )

    return embeddings


# --------------------------------------------------
# Single-document ingestion
# --------------------------------------------------

def _ingest_one(
    processor,
    src: Path,
    embed_fn,
    upsert_fn,
    RetrievedChunk,
) -> int:

    chunks_meta = processor.process_document(str(src))

    if not chunks_meta:
        raise ValueError(
            f"Document processor returned zero chunks: {src.name}"
        )

    # Use the complete relative path to prevent filename collisions.
    source_path = src.relative_to(DOCS_DIR).as_posix()

    chunks = [
        RetrievedChunk(
            text=item["text"],
            source=source_path,
            page_number=item.get("page_number"),
        )
        for item in chunks_meta
    ]

    texts = [chunk.text for chunk in chunks]

    logger.info(
        "Document {} produced {} chunks",
        source_path,
        len(chunks),
    )

    embeddings = _embed_document(embed_fn, texts)

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Final chunk and embedding counts do not match"
        )

    # Vector store upserts current chunks and removes obsolete
    # chunks for this source only after successful upload.
    upsert_fn(chunks, embeddings)

    return len(chunks)


# --------------------------------------------------
# Resumable document ingestion
# --------------------------------------------------

def seed_docs() -> dict:
    from app.models import RetrievedChunk
    from app.services.document_processor import DocumentProcessor
    from app.services.embedding_service import embed_texts
    from app.services.vector_store import upsert_chunks

    true_files = _select_corpus()
    manifest = _load_manifest()

    manifest["version"] = INGESTION_VERSION

    logger.info("=" * 60)
    logger.info("RESUMABLE DOCUMENT INGESTION")
    logger.info("Trusted documents: {}", len(true_files))
    logger.info("Corpus: seed/docs/true_data")
    logger.info("Manifest: {}", MANIFEST_PATH)
    logger.info("=" * 60)

    if not true_files:
        logger.warning("No true_data documents found")

        return {
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
            "chunks": 0,
        }

    processor = DocumentProcessor()

    counters = {
        "ingested": 0,
        "skipped": 0,
        "failed": 0,
        "chunks": 0,
    }

    start_time = time.time()
    total = len(true_files)

    for index, src in enumerate(true_files, start=1):

        relative_path = src.relative_to(DOCS_DIR).as_posix()
        fingerprint = _fingerprint(src)

        previous = manifest["documents"].get(
            relative_path,
            {},
        )

        # Skip unchanged documents that were successfully ingested.
        if (
            previous.get("status") == "completed"
            and previous.get("fingerprint") == fingerprint
            and previous.get("ingestion_version") == INGESTION_VERSION
        ):
            counters["skipped"] += 1

            logger.info(
                "[{}/{}] SKIPPED — already completed: {}",
                index,
                total,
                relative_path,
            )

            continue

        logger.info(
            "[{}/{}] PROCESSING: {}",
            index,
            total,
            relative_path,
        )

        entry = {
            "status": "processing",
            "fingerprint": fingerprint,
            "ingestion_version": INGESTION_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        manifest["documents"][relative_path] = entry

        # Save processing status before starting document ingestion.
        _save_manifest(manifest)

        try:
            chunk_count = _ingest_one(
                processor=processor,
                src=src,
                embed_fn=embed_texts,
                upsert_fn=upsert_chunks,
                RetrievedChunk=RetrievedChunk,
            )

            entry.update({
                "status": "completed",
                "chunk_count": chunk_count,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            })

            counters["ingested"] += 1
            counters["chunks"] += chunk_count

            logger.success(
                "[{}/{}] COMPLETED: {} — {} chunks",
                index,
                total,
                relative_path,
                chunk_count,
            )

        except Exception as exc:
            entry.update({
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

            counters["failed"] += 1

            logger.exception(
                "[{}/{}] FAILED: {}",
                index,
                total,
                relative_path,
            )

        finally:
            manifest["documents"][relative_path] = entry
            _save_manifest(manifest)

    elapsed = (time.time() - start_time) / 60

    logger.info("=" * 60)
    logger.info("INGESTION RUN FINISHED")
    logger.info("Successfully ingested: {}", counters["ingested"])
    logger.info("Already completed: {}", counters["skipped"])
    logger.info("Failed: {}", counters["failed"])
    logger.info("New chunks upserted: {}", counters["chunks"])
    logger.info("Time taken: {:.2f} minutes", elapsed)
    logger.info("=" * 60)

    return counters


# --------------------------------------------------
# Optional PostgreSQL database seeding
# --------------------------------------------------

def run_migrations(conn) -> None:
    cur = conn.cursor()

    try:
        files = sorted(
            path
            for path in MIGRATIONS_DIR.iterdir()
            if path.suffix.lower() == ".sql"
        )

        for path in files:
            logger.warning(
                "Running migration: {}",
                path.name,
            )

            cur.execute(
                path.read_text(encoding="utf-8")
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()


def seed_users(conn) -> None:
    cur = conn.cursor()

    try:
        for username, password, is_admin in DEMO_USERS:
            password_hash = hash_password(password)

            cur.execute(
                """
                INSERT INTO users (
                    username,
                    password_hash,
                    is_admin
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (username) DO UPDATE SET
                    password_hash = EXCLUDED.password_hash,
                    is_admin = EXCLUDED.is_admin
                """,
                (username, password_hash, is_admin),
            )

            logger.info(
                "Seeded user: {} (admin={})",
                username,
                is_admin,
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()


# --------------------------------------------------
# Main entry point
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable Kubernetes document ingestion"
    )

    parser.add_argument(
        "--seed-db",
        action="store_true",
        help=(
            "Run SQL migrations and seed demo users. "
            "WARNING: migrations may reset operational tables."
        ),
    )

    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip document ingestion.",
    )

    args = parser.parse_args()

    # Database migrations are never executed by default.
    if args.seed_db:
        logger.warning(
            "Explicit database seeding requested. "
            "Review SQL migrations before proceeding."
        )

        conn = psycopg2.connect(DATABASE_URL)

        try:
            run_migrations(conn)
            seed_users(conn)
        finally:
            conn.close()

        logger.success("Database seeding completed.")

    if args.no_ingest:
        logger.info("Document ingestion skipped.")
        return

    seed_docs()


if __name__ == "__main__":
    main()

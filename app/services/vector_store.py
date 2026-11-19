
from __future__ import annotations

import uuid

from langsmith import traceable
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.models import RetrievedChunk


VECTOR_SIZE = 1536
SCROLL_BATCH_SIZE = 256


def get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=5,
    )


def ensure_collection() -> None:
    client = get_client()

    existing = {
        collection.name
        for collection in client.get_collections().collections
    }

    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


def _get_source_point_ids(source: str) -> set[str]:
    """Retrieve all existing point IDs belonging to one document."""

    client = get_client()
    point_ids: set[str] = set()
    offset = None

    source_filter = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=source),
            )
        ]
    )

    while True:
        points, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=source_filter,
            limit=SCROLL_BATCH_SIZE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        point_ids.update(str(point.id) for point in points)

        if next_offset is None:
            break

        offset = next_offset

    return point_ids


def _delete_obsolete_source_chunks(
    source: str,
    current_point_ids: set[str],
) -> int:
    """
    Delete outdated chunks for a source while preserving
    all points belonging to its latest successful ingestion.
    """

    existing_ids = _get_source_point_ids(source)

    obsolete_ids = existing_ids - current_point_ids

    if not obsolete_ids:
        return 0

    client = get_client()

    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=PointIdsList(
            points=list(obsolete_ids)
        ),
        wait=True,
    )

    return len(obsolete_ids)


def upsert_chunks(
    chunks: list[RetrievedChunk],
    embeddings: list[list[float]],
) -> None:

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match"
        )

    if not chunks:
        return

    if any(len(vector) != VECTOR_SIZE for vector in embeddings):
        raise ValueError(
            f"All embeddings must have dimension {VECTOR_SIZE}"
        )

    sources = {chunk.source for chunk in chunks}

    if len(sources) != 1:
        raise ValueError(
            "upsert_chunks expects chunks from exactly one document"
        )

    ensure_collection()

    source = chunks[0].source
    client = get_client()

    points = []
    current_point_ids = set()

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings, strict=True)
    ):

        stable_key = f"{source}:{index}:{chunk.text}"

        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                stable_key,
            )
        )

        current_point_ids.add(point_id)

        payload = {
            "text": chunk.text,
            "source": source,
            "chunk_index": index,
        }

        if chunk.page_number is not None:
            payload["page_number"] = chunk.page_number

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )

    # First upload all current document chunks.
    # Do not delete old chunks before this succeeds.
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
        wait=True,
    )

    # Only after successful upload, remove obsolete chunks
    # belonging to this same document.
    deleted_count = _delete_obsolete_source_chunks(
        source=source,
        current_point_ids=current_point_ids,
    )

    if deleted_count:
        logger_message = (
            f"Removed {deleted_count} obsolete chunks "
            f"from {source}"
        )
        import logging
        logging.getLogger(__name__).info(logger_message)


@traceable(name="Qdrant_Dense_Search", run_type="retriever")
def search(
    query_embedding: list[float],
    top_k: int = 5,
) -> list[RetrievedChunk]:

    client = get_client()

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        RetrievedChunk(
            text=p.payload.get("text", ""),
            source=p.payload.get("source", ""),
            page_number=p.payload.get("page_number"),
            score=float(p.score),
        )
        for p in results
    ]


def _build_sparse_index():
    from app.services.sparse_vector_service import SparseVectorIndex

    client = get_client()

    all_points, _next_page = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    documents = [
        {
            "text": point.payload.get("text", "") if point.payload else "",
            "source": point.payload.get("source", "") if point.payload else "",
            "id": str(point.id),
        }
        for point in all_points
    ]

    sparse_index = SparseVectorIndex()
    sparse_index.fit(documents)

    return sparse_index


@traceable(name="TFIDF_Sparse_Search", run_type="retriever")
def sparse_search(
    query_text: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Pure sparse search using TF-IDF."""

    sparse_index = _build_sparse_index()

    return sparse_index.search(
        query_text,
        top_k=top_k,
    )


@traceable(name="Hybrid_RRF_Search", run_type="retriever")
def hybrid_search(
    query_embedding: list[float],
    query_text: str,
    top_k: int = 5,
    rrf_k: int = 60,
    sparse_top_k: int = 20,
) -> list[RetrievedChunk]:

    from app.services.sparse_vector_service import fuse_rrf

    dense_results = search(
        query_embedding,
        top_k=sparse_top_k,
    )

    sparse_index = _build_sparse_index()

    sparse_results = sparse_index.search(
        query_text,
        top_k=sparse_top_k,
    )

    fused = fuse_rrf(
        [dense_results, sparse_results],
        rrf_k=rrf_k,
    )

    return fused[:top_k]

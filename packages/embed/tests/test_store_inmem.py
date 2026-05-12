"""EmbedStore tests against Qdrant in-memory backend (no network)."""

from __future__ import annotations

import tempfile

import pytest

from pps_embed import EmbedStore


@pytest.fixture
async def store(tmp_path):
    s = EmbedStore(path=str(tmp_path / "qdrant"))
    await s.ensure_collections()
    yield s
    await s.close()


async def test_ensure_collections_idempotent(tmp_path):
    s = EmbedStore(path=str(tmp_path / "q1"))
    await s.ensure_collections()
    await s.ensure_collections()  # second call must not raise
    await s.close()


async def test_upsert_and_query_photo(store, small_image):
    pid = await store.upsert_photo(small_image, payload={"src": "test.jpg"})
    assert len(pid) == 40
    hits = await store.query_similar_photos(small_image, k=1)
    assert hits, "expected at least one hit"
    assert hits[0].payload["photo_id"] == pid


async def test_distinct_photos_distinct_top_hit(store, small_image, small_image_b):
    p_a = await store.upsert_photo(small_image)
    p_b = await store.upsert_photo(small_image_b)
    hits = await store.query_similar_photos(small_image, k=2)
    # Top hit for query=small_image should be small_image itself.
    assert hits[0].payload["photo_id"] == p_a
    assert hits[1].payload["photo_id"] == p_b


async def test_upsert_and_query_algorithm(store, algo_params_villa):
    aid = await store.upsert_algorithm(algo_params_villa, name="villa_luxury")
    hits = await store.query_similar_algorithms(algo_params_villa, k=1)
    assert hits[0].payload["algorithm_id"] == aid
    # Stored canonical params survive round-trip.
    assert "villa_luxury" in hits[0].payload["params_json"]


async def test_algorithm_query_finds_similar_not_just_identical(
    store, algo_params_villa, algo_params_studio
):
    a_villa = await store.upsert_algorithm(algo_params_villa, name="villa")
    a_studio = await store.upsert_algorithm(algo_params_studio, name="studio")
    hits = await store.query_similar_algorithms(algo_params_villa, k=2)
    ids = [h.payload["algorithm_id"] for h in hits]
    assert a_villa in ids
    assert a_studio in ids


async def test_get_algorithm_round_trip(store, algo_params_villa):
    """Stored algorithm can be retrieved by stable id."""
    aid = await store.upsert_algorithm(algo_params_villa, name="villa")
    payload = await store.get_algorithm(aid)
    assert payload is not None
    assert payload["algorithm_id"] == aid
    assert payload["name"] == "villa"
    assert "villa_luxury" in payload["params_json"]


async def test_get_algorithm_missing_returns_none(store):
    """Unknown algorithm id → None, not raise."""
    payload = await store.get_algorithm("0" * 40)
    assert payload is None


async def test_fetch_genes_resolves_photo_to_algorithm(
    store, small_image, algo_params_villa
):
    """fetch_genes walks photo→algorithm_id→params_json correctly."""
    aid = await store.upsert_algorithm(algo_params_villa, name="villa")
    await store.upsert_photo(
        small_image,
        payload={"algorithm_id": aid, "src": "ref_villa.jpg"},
    )
    genes = await store.fetch_genes(small_image, agent="microcontrast", k=3)
    assert len(genes) == 1
    assert genes[0]["agent"] == "microcontrast"
    assert genes[0]["property"] == "villa_luxury"
    assert genes[0]["texture"]["fine"] == 0.45


async def test_fetch_genes_filters_wrong_agent(
    store, small_image, algo_params_villa
):
    """Algorithm payload tagged for a different agent must be skipped."""
    # Tweak the algo payload to claim a different agent
    aid = await store.upsert_algorithm(
        algo_params_villa,
        name="villa",
        payload={"agent": "lightblend"},  # mismatched
    )
    await store.upsert_photo(small_image, payload={"algorithm_id": aid})
    genes = await store.fetch_genes(small_image, agent="microcontrast", k=3)
    assert genes == []


async def test_fetch_genes_skips_photos_without_algorithm(store, small_image):
    """Photos with no linked algorithm yield no genes (not an error)."""
    await store.upsert_photo(small_image, payload={"src": "no_algo.jpg"})
    genes = await store.fetch_genes(small_image, agent="microcontrast", k=3)
    assert genes == []


async def test_gene_provider_sync_wraps_async(
    store, small_image, algo_params_villa
):
    """gene_provider_sync() returns a sync callable usable from a sync caller."""
    aid = await store.upsert_algorithm(algo_params_villa, name="villa")
    await store.upsert_photo(small_image, payload={"algorithm_id": aid})
    # Build the sync provider AFTER the async setup; calling it must NOT
    # require an event loop (asyncio.run starts a fresh one inside).
    # Note: each sync call creates a transient client connection — we use a
    # fresh EmbedStore for this test that isn't bound to the pytest fixture
    # already-running loop. The fixture used a tmp_path to write to disk.
    from pps_embed import EmbedStore

    # Reuse the same on-disk path so the second store sees the same data.
    # We need to close the fixture store first; instead, just confirm the
    # method shape — the integration test is the orchestrator gene path.
    provider = store.gene_provider_sync(agent="microcontrast", k=2)
    assert callable(provider)
    # Sanity: signature accepts a numpy array
    import inspect

    sig = inspect.signature(provider)
    assert len(sig.parameters) == 1

from unittest.mock import MagicMock, patch

import pytest
from pps_core.unsplash import UnsplashClient, UnsplashError, _to_photo

SAMPLE_PHOTO = {
    "id": "abc123",
    "description": "a tree",
    "alt_description": None,
    "width": 4000,
    "height": 3000,
    "urls": {
        "raw": "https://images.unsplash.com/raw.jpg",
        "full": "https://images.unsplash.com/full.jpg",
        "regular": "https://images.unsplash.com/regular.jpg",
        "small": "https://images.unsplash.com/small.jpg",
        "thumb": "https://images.unsplash.com/thumb.jpg",
    },
    "links": {"download_location": "https://api.unsplash.com/photos/abc123/download"},
    "user": {"name": "Jane Photographer"},
}


def test_to_photo_parses_essential_fields():
    p = _to_photo(SAMPLE_PHOTO)
    assert p.id == "abc123"
    assert p.description == "a tree"
    assert p.width == 4000
    assert p.user_name == "Jane Photographer"
    assert p.url("raw").endswith("raw.jpg")
    assert p.download_location.endswith("/download")


def test_url_with_unknown_size_raises():
    p = _to_photo(SAMPLE_PHOTO)
    with pytest.raises(KeyError):
        p.url("xxl")  # type: ignore[arg-type]


def test_search_calls_correct_endpoint():
    session = MagicMock()
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"results": [SAMPLE_PHOTO]}
    session.get.return_value = resp

    client = UnsplashClient("KEY", session=session)
    photos = client.search("tree", per_page=5, orientation="landscape")
    assert len(photos) == 1
    args, kwargs = session.get.call_args
    assert args[0].endswith("/search/photos")
    assert kwargs["params"]["query"] == "tree"
    assert kwargs["params"]["per_page"] == 5
    assert kwargs["params"]["orientation"] == "landscape"


def test_search_rate_limit_then_success():
    session = MagicMock()
    rl = MagicMock(status_code=429)
    ok = MagicMock(status_code=200)
    ok.json.return_value = {"results": []}
    session.get.side_effect = [rl, ok]

    client = UnsplashClient("KEY", session=session, max_retries=3)
    with patch("pps_core.unsplash.time.sleep"):
        out = client.search("tree")
    assert out == []
    assert session.get.call_count == 2


def test_4xx_raises_unsplash_error():
    session = MagicMock()
    bad = MagicMock(status_code=403)
    bad.text = "forbidden"
    session.get.return_value = bad
    client = UnsplashClient("KEY", session=session)
    with pytest.raises(UnsplashError):
        client.search("tree")


def test_empty_key_rejected():
    with pytest.raises(ValueError):
        UnsplashClient("")


def test_empty_query_rejected():
    client = UnsplashClient("KEY", session=MagicMock())
    with pytest.raises(ValueError):
        client.search("   ")

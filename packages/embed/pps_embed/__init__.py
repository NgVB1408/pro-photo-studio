"""Photo + algorithm embeddings → Qdrant; metadata → Postgres."""

from .algo_features import algo_embedding, canonicalise_params
from .photo_features import PHOTO_DIM, photo_embedding
from .schema import Algorithm, AuditLog, DatasetEntry, Embedding, Photo
from .store import EmbedStore, QueryHit

__all__ = [
    "Algorithm",
    "AuditLog",
    "DatasetEntry",
    "Embedding",
    "EmbedStore",
    "PHOTO_DIM",
    "Photo",
    "QueryHit",
    "algo_embedding",
    "canonicalise_params",
    "photo_embedding",
]

__version__ = "0.1.0"

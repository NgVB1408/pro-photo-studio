"""User-facing Vietnamese strings for pps-embed CLI / errors."""

from __future__ import annotations

CLI_HELP_APP = "Pro Photo Studio — vector store cho photo + algorithm gene."
CLI_HELP_INDEX = "Index 1 ảnh vào Qdrant 'photos' collection."
CLI_HELP_QUERY = "Tìm top-K ảnh tương tự."
CLI_HELP_INDEX_ALGO = "Index 1 parameter set vào Qdrant 'algorithms' collection."
CLI_HELP_MIGRATE = "Áp Alembic migrations lên Postgres metadata DB."
ARG_MIGRATE_CHECK = (
    "Chỉ kiểm tra cú pháp migrations offline (không cần DB) — dùng cho CI."
)
ARG_DB_URL = "DATABASE_URL override (mặc định lấy từ env)."

ARG_IMAGE = "Đường dẫn ảnh (jpg/png/tiff)"
ARG_K = "Số kết quả trả về (top-K)"
ARG_NAME = "Tên dễ đọc cho parameter set"
ARG_PARAMS_JSON = "Đường dẫn file JSON chứa parameter set"
ARG_QDRANT_URL = "URL Qdrant (mặc định lấy từ env QDRANT_URL)"
ARG_QDRANT_KEY = "API key Qdrant (mặc định lấy từ env QDRANT_API_KEY)"

ERR_QDRANT_NO_URL = (
    "Chưa cấu hình QDRANT_URL. Set env QDRANT_URL=https://<host>:6333 hoặc "
    "truyền --qdrant-url."
)
ERR_NO_DB_URL = (
    "Chưa cấu hình DATABASE_URL. Set env DATABASE_URL="
    "postgresql+asyncpg://user:pass@host/db."
)
ERR_NOT_AN_IMAGE = "Không đọc được ảnh: {path}"

INFO_INDEXED = "Đã index ảnh, photo_id={pid}"
INFO_INDEXED_ALGO = "Đã index algorithm, algorithm_id={aid}"
INFO_QUERY_HEADER = "Top {k} kết quả gần nhất:"
INFO_MIGRATE_DONE = "Migrate xong — schema đã sẵn sàng."
INFO_MIGRATE_CHECK = "Migrations OK ({n} revision, head={heads}). Không kết nối DB."
ERR_MIGRATE_CHECK_FAILED = "Migrations không hợp lệ: {err}"

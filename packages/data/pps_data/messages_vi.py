"""User-facing Vietnamese strings for the CLI + error messages.

Code, logger output, and exception class names stay English so SO/Google still
work. Only the strings the *end user* reads are translated.
"""

from __future__ import annotations

CLI_HELP_APP = "Pro Photo Studio — công cụ stream dataset (FiveK / LSD / SUN)."
CLI_HELP_SAMPLE = "Lấy mẫu N ảnh từ dataset, lưu ra thư mục output."
CLI_HELP_INSPECT = "Mở subset trên FiftyOne để soi visual."
CLI_HELP_LIST = "Liệt kê dataset hỗ trợ + repo HF mirror đang dùng."

ARG_DATASET = "Tên dataset: fivek / lsd / sun"
ARG_N = "Số lượng mẫu cần lấy"
ARG_OUT = "Thư mục output (sẽ tạo nếu chưa có)"
ARG_EXPERT = "Expert FiveK (a-e). Mặc định 'c' — đáp án chuẩn cho fine-tune"
ARG_SPLIT = "HF split. Hầu hết dataset chỉ có 'train'"
ARG_MIRROR = "Override HF repo id (mặc định lấy từ env hoặc default)"

ERR_NO_HF_TOKEN = (
    "Chưa có HF_TOKEN. Set biến môi trường HF_TOKEN bằng access-token "
    "đọc-only của bạn (https://huggingface.co/settings/tokens)."
)
ERR_DATASET_UNKNOWN = "Dataset '{name}' chưa hỗ trợ. Có: {options}."
ERR_FIFTYONE_MISSING = (
    "FiftyOne chưa cài. Chạy: pip install 'pps-data[fiftyone]' rồi thử lại."
)

INFO_SAMPLE_DONE = "Đã lưu {n} mẫu vào {out}."
INFO_INSPECT_OPEN = "Mở FiftyOne tại http://localhost:5151 (Ctrl+C để thoát)."
INFO_NO_SAMPLES = "Không lấy được mẫu nào — kiểm tra HF_TOKEN hoặc kết nối mạng."

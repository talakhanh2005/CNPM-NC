# Tích hợp AI với backend

API tương thích `face-emotion-backend/app/integrations/ai.py` và
`app/integrations/ai_schema.py`. Backend không cần đổi request/response.
Các hàm Python `process_realtime_batch`, `process_video_meeting` cũ vẫn phục vụ
script nội bộ; HTTP API dùng `AIService.analyze_frame` và `analyze_video`.

## Chạy trên Windows

Từ thư mục có `pyproject.toml`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Đặt `AI_API_KEY` trong `.env` thành cùng giá trị với backend. Đặt trọng số
EfficientNet-B0 đã huấn luyện tại `models/emotion_model_best.pth`, hoặc sửa
`AI_MODEL_PATH` thành đường dẫn thực tế. Đây phải là state dict có đầu ra 7 lớp
theo thứ tự `angry, disgust, fear, happy, neutral, sad, surprise`.
API dừng khởi động nếu thiếu/sai trọng số; không tự tạo dự đoán ngẫu nhiên.

```powershell
.\.venv\Scripts\python.exe -m uvicorn openemotion.api.app:create_app --factory --app-dir src --host 127.0.0.1 --port 9000
```

Backend cần cấu hình:

```dotenv
AI_MODE=http
AI_BASE_URL=http://localhost:9000
AI_API_KEY=<cùng khóa với AI>
AI_TIMEOUT_SECONDS=120
FRAME_TIMEOUT_SECONDS=10
```

Chạy cả API backend và worker backend (`python -m app.worker`) để job video được
xử lý. Production dùng HTTPS theo quy tắc cấu hình của backend. Khi chạy trên máy
khác/container, thay hostname bằng địa chỉ AI mà backend truy cập được.

`GET /health` chỉ trả thành công sau khi model đã load. Các endpoint phân tích
yêu cầu `Authorization: Bearer <AI_API_KEY>`.

## Realtime

`POST /v1/analyze/frame` nhận multipart: `frame` JPEG/PNG, `frame_id` chuỗi,
`timestamp` ISO 8601 có timezone. Trả trực tiếp `FrameResult`, echo metadata.

- Chọn mặt lớn nhất; trả đủ 7 xác suất, confidence bằng xác suất cao nhất.
- Ánh xạ `disgust → disgusted`, `fear → fearful`, `surprise → surprised` chỉ tại
  adapter; không đổi thứ tự output model hoặc nhãn dataset.
- Không thấy mặt: HTTP 200, `face_detected=false`, `emotion=fail_detection`,
  `failure_reason=no_face`, confidence và mọi xác suất bằng 0, `face_id=null`.
- Ảnh lỗi: HTTP 422; lỗi suy luận: HTTP 500. Backend chuyển lỗi HTTP thành
  `failure_reason=ai_error` theo logic hiện tại.
- `face_id` là số thứ tự phát hiện trong frame chuyển thành chuỗi, không phải
  danh tính hay ID theo dõi xuyên các frame. Student/meeting do backend xác định.
- Giới hạn mặc định 512 KiB và 1920×1080 pixel. Frame chỉ nằm trong RAM.

## Video

`POST /v1/analyze/video` nhận JSON `{recording_id, video_url, duration}` và header
`Idempotency-Key` là ID job của backend. Trả HTTP 200 trực tiếp `BatchResult`,
không trả 202/job_id và không bọc envelope.

- Tải ngay URL có chữ ký về thư mục tạm. Hỗ trợ endpoint Cloudinary
  `/video/download?format=...` và redirect đến host được phép. HTTPS, giới hạn
  byte và thời gian được kiểm tra. File tạm được xóa cả khi thành công hoặc lỗi.
- `AI_DOWNLOAD_HOSTS` mặc định là `api.cloudinary.com`, `res.cloudinary.com`.
  Thêm chính xác host CDN đáng tin cậy khi dùng cấu hình Cloudinary khác.
- Lấy 1 frame/giây, mỗi khuôn mặt hợp lệ là một mẫu. Đếm toàn bộ mẫu thành số
  nguyên; phần trăm do backend tính. Không suy đoán `student_id` từ `face_id`.
- Timeline chứa timestamp giây và confidence của từng mẫu. Tối đa 20.000 điểm
  trải đều trên danh sách mẫu; số đếm vẫn đầy đủ. Timestamp vượt duration do BE
  gửi sẽ bị từ chối. Không thấy mặt trả `{sample_counts: {}, timeline: []}`.
- SQLite tại `AI_CACHE_PATH` lưu kết quả thành công theo idempotency key, tồn tại
  qua restart. Retry với URL ký mới vẫn trả kết quả cũ cho cùng recording/duration.
  Dùng cùng key cho recording/duration khác: HTTP 409. Lỗi cho phép thử lại cùng key.
  Job đang chạy: HTTP 409; job bị crash có thể được nhận lại sau khi lease hết hạn.
  Worker cũ không được ghi đè kết quả của lease mới. Giữ file SQLite trên ổ bền vững.
- Mặc định ngân sách xử lý video 110 giây, dưới timeout BE 120 giây. Pipeline kiểm
  tra deadline giữa các thao tác/frame; không cưỡng bức ngắt một lệnh suy luận native
  đang chạy. Video dài/CPU chậm có thể timeout: tăng timeout cả hai bên và
  `JOB_LEASE_SECONDS` của backend tương ứng, hoặc dùng GPU.
- HTTP download lỗi: 502; timeout: 504; video/schema sai: 422; suy luận lỗi: 500.
  Response/log không chứa URL ký hay nội dung lỗi bên trong.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Test tích hợp gọi HTTP API bằng chính `HttpAIClient` và schema của backend khi repo
backend nằm cạnh repo AI theo cấu trúc workspace. Có thể đặt `BACKEND_ROOT` thành
thư mục backend chứa `app/` ở máy khác. Nếu thiếu backend, test đó được đánh dấu skip.
Test API dùng logits cố định; vẫn chạy decode JPEG/MP4, tiền xử lý, softmax,
adapter, download, cache và HTTP thật trong ASGI. Test model huấn luyện được đánh
dấu skip khi thiếu file trọng số; không coi test logits cố định là kiểm định độ chính xác AI.

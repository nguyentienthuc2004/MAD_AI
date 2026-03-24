# AI Image Moderation Service

Dịch vụ FastAPI dùng để kiểm tra mức độ nhạy cảm của ảnh (NSFW) bằng mô hình đã fine-tune trong thư mục `checkpoint`.

## Chức năng

- Nhận nhiều ảnh trong một request.
- Trả về nhãn dự đoán, điểm NSFW và kết luận nhạy cảm theo ngưỡng `threshold`.
- Có endpoint health check để kiểm tra trạng thái model.

## Cấu trúc chính

- `app.py`: API FastAPI và logic suy luận.
- `checkpoint/`: model và processor đã huấn luyện.
- `requirements.txt`: danh sách thư viện cần cài.

## Yêu cầu

- Python 3.10 trở lên.
- Có thư mục `checkpoint` hợp lệ.

## Cài đặt

1. Tạo môi trường ảo:

```bash
python -m venv venv
```

2. Kích hoạt môi trường ảo:

```bash
# Windows (Git Bash)
source venv/Scripts/activate

# Linux/macOS
source venv/bin/activate
```

3. Cài thư viện:

```bash
pip install -r requirements.txt
```

## Chạy server

Chạy trực tiếp bằng file `app.py`:

```bash
python app.py
```

Mặc định server chạy tại:

```bash
http://0.0.0.0:3001
```

Đổi host/port bằng biến môi trường:

```bash
# Linux/macOS
export HOST=0.0.0.0
export PORT=3001
python app.py
```

```bash
# Windows CMD
set HOST=0.0.0.0
set PORT=3001
python app.py
```

## API

1. Health check

`GET /health`

Response gồm:
- `ok`: model load thành công hay không.
- `checkpointDir`: đường dẫn checkpoint đang dùng.
- `error`: nội dung lỗi nếu model load thất bại.

2. Moderate images

`POST /moderate-images`  
`Content-Type: multipart/form-data`

Fields:
- `files`: danh sách file ảnh.
- `threshold`: tùy chọn, số thực trong khoảng từ 0 đến 1 (mặc định 0.70).

Response:
- `success`
- `threshold`
- `isSensitive`
- `flags`
- `results`: danh sách kết quả theo từng ảnh.

## Ví dụ gọi API bằng curl

```bash
curl -X POST "http://127.0.0.1:3001/moderate-images" \
   -F "files=@test1.jpg" \
   -F "files=@test2.png" \
   -F "threshold=0.7"
```

## Lỗi thường gặp

1. Chạy `python app.py` rồi thoát ngay
- Nguyên nhân cũ: chưa có khối khởi động server trong `app.py`.
- Hiện tại đã có, chỉ cần chạy lại lệnh.

2. Báo lỗi cổng `3001` đã được sử dụng
- Đóng tiến trình đang chiếm cổng `3001`, hoặc đổi `PORT` sang cổng khác.

3. Cảnh báo `torchvision WinError 127`
- Đây thường là cảnh báo extension image của `torchvision`.
- Nếu API vẫn chạy và suy luận bình thường thì có thể tiếp tục sử dụng.

## Ghi chú

- Ngưỡng NSFW mặc định lấy từ biến `NSFW_THRESHOLD` (0.70).
- Đường dẫn model mặc định là thư mục `checkpoint` cùng cấp với `app.py`.

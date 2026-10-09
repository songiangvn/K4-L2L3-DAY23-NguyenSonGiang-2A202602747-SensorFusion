# Bonus CVAT — cách import và chụp ảnh bằng chứng

Track của lần chạy chấm điểm (segment mặc định) **không có ghost và không đổi ID** (`cvat_id_events.json`),
nên bằng chứng CVAT dùng segment thứ 4 của khoá học
`training_segment-10963653239323173269_1924_000_1944_000` (tag `_109636`), cùng code E–H, seed 0, frame 0–198.

## File

| File | Nội dung |
|---|---|
| `cvat_tracks_fused.json`, `cvat_tracks_fused_109636.json` | Export qua `fusion_lab.export_cvat.export_tracks_json` (mọi track mỗi frame: id, state, score, x, v, kích thước, yaw) |
| `cvat_annotations_109636.xml` | CVAT for video 1.1: box 2D = hình chiếu hộp 3D của track confirmed lên camera FRONT, thuộc tính `track_id` = ID của tracker |
| `cvat_id_events_109636.json` | Danh sách đổi ID / ghost tìm tự động (ghép 2 m XY như platform) |
| `cvat_preview_109636.png` | Preview (vẽ lại từ XML) các frame 67, 73, 91, 137 |

Ảnh FRONT (`data/bonus_cvat/images_109636/frame_000000.jpg` … `frame_000198.jpg`) **không commit** (dữ liệu Waymo).

## Các bước (≈ 10 phút)

1. Mở CVAT (app.cvat.ai hoặc CVAT local) → **Tasks → +** → tên `day23_109636`.
2. **Labels → Add label**: tên `vehicle_track`; thêm attribute `track_id`, kiểu **Number** (0;100000;1).
3. **Select files**: chọn toàn bộ 199 ảnh trong `data/bonus_cvat/images_109636/` → **Submit & Open**.
4. Trong task: **Actions → Upload annotations** → format **CVAT 1.1** → chọn `student/bonus/cvat_annotations_109636.xml`.
5. Mở job, chụp màn hình tại:
   - **frame 67** (xe bạc giữa làn: `track_id = 16`) và **frame 73** (cùng xe: `track_id = 18`) → **đổi ID**;
   - **frame 91** (cùng xe: `track_id = 20`) → đổi ID lần 2;
   - **frame 137** (`track_id = 20`, xe ở ~50 m) → **ghost** theo metric (tâm GT đã ra ngoài cửa sổ 50 m).
6. Lưu ảnh chụp vào `student/bonus/cvat_screenshot_*.png` rồi commit. (Đã làm: `cvat_screenshot_frame067/073/091/137.png`.)

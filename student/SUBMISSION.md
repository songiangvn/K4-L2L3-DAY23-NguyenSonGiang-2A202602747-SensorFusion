# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyễn Sơn Giang
- MSSV: 2A202602747
- Email: songiangvn@gmail.com
- Link repo (fork): https://github.com/songiangvn/K4-L2L3-DAY23-NguyenSonGiang-2A202602747-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): commit cuối trên `main` của fork; hash 40 ký tự được nộp trên LMS (một commit không thể tự ghi hash của chính nó)

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]` (199 frame), `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `seed = 0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision 0.9701, recall 0.7004, tp 519 / fp 16 / fn 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1503 m, 502, 11.3436 m², 0, 239, 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1359 m, 502, 9.2668 m², 0, 239, 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: xem bảng và phân tích dưới đây.

| Mode | RMSE (m) | matches | sum_sq_err (m²) | ghost | miss | precision_track | coverage |
|---|---|---|---|---|---|---|---|
| lidar | 0.1503 | 502 | 11.3436 | 0 | 239 | 502/502 = 1.000 | 502/519 = 0.967 |
| fused | 0.1359 | 502 | 9.2668 | 0 | 239 | 502/502 = 1.000 | 502/519 = 0.967 |

`rmse_fused − rmse_lidar = −0.0145 m` (≤ 0.05 m: camera không làm tracking xấu đi).

**So sánh công bằng vì mẫu số giống hệt nhau.** Hai mode có cùng 502 cặp ghép, 0 ghost, 239 miss. Hai RMSE vì vậy được tính trên đúng cùng tập cặp track–xe, và phần giảm RMSE đến từ sai số vị trí nhỏ hơn chứ không phải do ghép ít cặp hơn. Tổng bình phương sai số giảm từ 11.34 xuống 9.27 m² (−18%).

Đối chiếu từng frame trong `grade_run_lidar.log` và `grade_run_fused.log`:
- Cả 199 record có `det_tp/det_fp/det_fn`, `valid_gt`, `confirmed`, `matches`, `ghosts`, `misses` **giống hệt nhau**; chỉ `sum_sq_err` khác.
- Fused có `sum_sq_err` nhỏ hơn ở 137/195 frame có cặp ghép.

**Ghost = 0.** Detector có precision cao (16 FP trên 199 frame). Một FP lẻ chỉ tạo track `initialized`/`tentative` với score 1/6, rồi bị xoá ở lượt LiDAR kế tiếp. Muốn thành confirmed phải đạt score > 0.8, tức có hit ở 5 frame liên tiếp.

**Miss = 239.** Miss chủ yếu do detector, không phải do tracker:
- 222 trong 239 miss là xe detector không phát hiện được (`det_fn = 222`). Vì vậy `coverage` chỉ tính trên `det_tp`.
- 17 miss còn lại (`det_tp − matches = 519 − 502`) chủ yếu là độ trễ xác nhận. Ở frame 0–3, 2 xe đã được detect nhưng track mới đạt score 4/6 < 0.8 nên chưa confirmed (8 miss). Frame 44–47 có một xe mới vào và cũng trải qua giai đoạn này.
- Ngược lại, có 5 frame `matches > det_tp`: detector bỏ sót xe nhưng track confirmed vẫn được EKF predict đúng chỗ. Đây là lợi ích của tracking so với detection từng frame.

**Giới hạn.** Đo camera là tâm hộp 2D **ground-truth** của camera FRONT cộng nhiễu σ = 0.5 px, không phải output của một camera detector. Mức cải thiện 0.0145 m vì vậy là cận trên, trong điều kiện đo camera gần như hoàn hảo về nhiễu. Ngược lại, tâm hộp 2D không trùng hình chiếu của tâm hộp 3D, nên tạo bias có hệ thống (xem Bonus: γu trung bình ≈ −12 px khi calibration đúng). Kết quả không chứng minh chất lượng của một hệ perception độc lập với ground truth.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?**

   | | LiDAR | Camera |
   |---|---|---|
   | Đo `z` | `z = (x, y, z)` tâm hộp 3D, đơn vị mét, dim 3 | `z = (u, v)` pixel, dim 2 (`build_camera_measurement`) |
   | Nhiễu `R` | `diag(0.1², 0.1², 0.1²)` m² | `diag(σ_cam_i², σ_cam_j²) = diag(5², 5²)` px² (lấy từ `get_tracking_params()`) |
   | Mô hình đo `h(x)` | Tuyến tính: `h(x) = R_vs·p + t`, `H = [R_vs | 0]` (3×6) | Phi tuyến pinhole: `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s` (`camera_measurement_prediction`) |
   | Jacobian | `H` không đổi | `H` là Jacobian tại `x` hiện tại (platform `get_H`), nên đây là **E**KF |

   Cùng một `ekf_update` (`kalman.py`) xử lý cả hai: `γ = z − h(x)`, `S = H P Hᵀ + R`, `K = P Hᵀ S⁻¹`, `x ← x + Kγ`, `P ← (I − KH)P`.
   - Camera không đo độ sâu (`H` có hạng 2): nó chủ yếu sửa vị trí **ngang/dọc trong ảnh**, tức `y` và `z` của xe phía trước. Trong bonus, camera update trên track 0 dịch `y` khoảng 0.05–0.24 m, còn `x` chỉ khoảng 0.01–0.05 m.
   - Đo camera còn chỉ có nghĩa khi điểm nằm trước camera, nên phải kiểm tra `x_s > 1e-6` trước khi chia cho `x_s`.

2. **Vì sao cần gating Mahalanobis trước khi gán?**
   - `d² = γᵀ S⁻¹ γ` (`mahalanobis_distance`) chuẩn hoá residual theo bất định `S`, gồm cả bất định của track (`HPHᵀ`) lẫn của đo (`R`). Khi gán đúng, `d²` theo phân phối χ² với bậc tự do bằng `dim_meas`. Cổng `chi2.ppf(0.995, dim)` vì vậy là 12.84 cho LiDAR (3D) và 10.60 cho camera (2D), xem `chi2_gate`.
   - Không có gating thì greedy (`pick_next_pair`) sẽ ghép mọi track với một đo nào đó, kể cả đo cách rất xa (FP, xe khác). EKF update khi đó kéo track lệch hẳn, hoặc làm đổi ID.
   - So với Euclidean: track mới có `P` vận tốc lớn (σ = 50 m/s), nên sau predict `P` vị trí lớn và cổng tự rộng ra. Track đã hội tụ có cổng hẹp. Một ngưỡng Euclidean cố định không phân biệt được hai trường hợp này. Hai chiều có bất định khác nhau (ví dụ dọc vs ngang hướng chuyển động) cũng được cân đúng theo `S`.
   - Pixel và mét có đơn vị khác nhau. `d²` không có đơn vị, nên một chuẩn duy nhất áp dụng được cho cả LiDAR lẫn camera.
   - `association_cost_matrix` kiểm tra `meas.sensor.in_fov(track.x)` **trước** khi tính `d²`. Track sau lưng camera (độ sâu ≤ 0) không bao giờ bị chiếu, vì phép chia cho `x_s` sẽ vô nghĩa hoặc gây lỗi.

3. **Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.**

   **Track-then-fuse.** Chỉ có một `TrackManager` và một danh sách track. Mỗi frame trong `run_lab.py` chạy theo thứ tự:
   - (1) `KF.predict` mọi track **một lần**;
   - (2) `associate_and_update(manager, lidar_obs, KF, lidar_sensor)` (AssocL): EKF update bằng LiDAR, cộng/trừ score, tạo/xoá track;
   - (3) chỉ khi `--fusion fused`: `associate_and_update(manager, camera_obs, KF, camera_sensor)` (AssocC), EKF update bằng camera trên **cùng các track đó**.

   Dữ liệu thô của hai cảm biến không bao giờ được gộp trước detection. Detection chỉ từ LiDAR BEV; camera vào ở mức **track**.

   Dẫn chứng trên log: so `grade_run_lidar.log` với `grade_run_fused.log` theo từng frame.
   - Mọi record có cùng `det_tp/det_fp/det_fn`: detection giống nhau, camera không đụng tới detection.
   - Mọi record có cùng `confirmed`, `matches`, `ghosts`, `misses`: tập track và vòng đời giống nhau, vì chỉ lượt LiDAR quyết định.
   - Chỉ `sum_sq_err` khác. Ví dụ frame 50: LiDAR 0.1230 m², fused 0.1143 m², cùng 3 confirmed, 3 matches.

   Như vậy camera chỉ tinh chỉnh **trạng thái** của track đã có.

4. **Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**
   - Lệch extrinsic làm `h(x)` chiếu sai chỗ một cách **có hệ thống**. Innovation `γ = z − h(x)` vì vậy không còn trung bình ≈ 0 mà có **bias** cùng dấu ở mọi frame và mọi track. Lệch yaw tạo bias chủ yếu theo `u` (ngang). Lệch tịnh tiến ngang tạo bias theo `u`, lớn hơn với xe gần (tỉ lệ 1/x_s).
   - NIS (`d²`) trung bình tăng vượt `dim = 2`. Nhiều đo rơi ra ngoài cổng χ², nên số lần camera update giảm mạnh.
   - Trạng thái bị kéo lệch ngang, dẫn đến RMSE tăng, và mức tăng phụ thuộc vào việc gating có chặn hay không.

   Đo thực nghiệm ở phần Bonus: lệch yaw 0.5° đưa γu trung bình từ −11.9 lên −26.9 px, NIS từ 2.25 lên 7.28, RMSE fused từ 0.136 lên 0.211 m. Lệch ≥ 1° thì gần như mọi đo bị cổng loại, chỉ còn 20 update thay vì 509, và RMSE quay về gần mức chỉ LiDAR.

5. **Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Vì sao lidar quyết định score/init/delete còn camera chỉ EKF update?**

   Sensor tường minh ở frame rỗng:
   - Khi detector không trả hộp nào, `meas_list = []`, nên không thể suy ra loại sensor từ `meas_list[0].sensor`. Nhưng lượt LiDAR rỗng vẫn phải chạy vòng đời: mọi track trong FOV LiDAR bị tính **miss** (−1/6 score) và track hết score bị xoá.
   - Nếu bỏ qua frame rỗng, track ma sẽ sống mãi. Nếu đoán nhầm là lượt camera, sẽ không có ai trừ score.
   - Vì vậy `associate_and_update` luôn gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)` với `sensor` được truyền vào. Test `test_empty_lidar_frame_scores_then_deletes_exhausted_track` kiểm tra đúng điều này.

   LiDAR quyết định vòng đời, camera chỉ update EKF:
   - LiDAR cho đo **3D đầy đủ** và phủ 360°, là cảm biến dùng để detect. Một hit LiDAR là bằng chứng trực tiếp xe tồn tại tại vị trí 3D đó.
   - Camera chỉ cho 2D, không có độ sâu, và chỉ thấy trong FOV FRONT. Một đo camera không đủ để khởi tạo track (không biết `x`). Việc camera thiếu đo cũng không chứng tỏ xe biến mất: xe có thể nằm ngoài FOV, bị che, hoặc frame không có nhãn FRONT.
   - Nếu camera cũng cộng/trừ score, một xe sẽ được tính hai lần mỗi frame trong FOV camera và chỉ một lần ngoài FOV. Ngưỡng `confirmed_threshold`/`delete_threshold` khi đó mất ý nghĩa, và lỗi camera (ví dụ lệch calibration làm mọi đo bị gate loại) sẽ xoá nhầm track thật.
   - Vì vậy `handle_updated_track` chỉ ghi hit khi `sensor.name == "lidar"`, và `manage_tracks` return ngay nếu không phải LiDAR.

6. **Điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xoá track** (`track_management.py`, `window = 6`):
   - **Khởi tạo** (`init_track_state_from_meas`): đổi `z` LiDAR sang vehicle frame (`sens_to_veh`). Đặt `x = [p; 0]`, `P_pos = M R Mᵀ` (R xoay theo `M`), `P_vel = diag(50², 50², 5²)`, `score = 1/6`, `state = "initialized"`.
   - **Score** (`update_track_score`, chỉ lượt LiDAR): hit thì `score = min(1, score + 1/6)`. Miss trong FOV LiDAR thì `score −= 1/6`; track ngoài FOV không bị trừ.
   - **Trạng thái:** track chưa confirmed nhận hit thì thành `tentative`. Khi `score > 0.8` (sau 5 hit liên tiếp: 1/6 → 5/6 ≈ 0.833) thì thành `confirmed`. Trên log, track đầu tiên confirmed ở frame 4.
   - **Giữ confirmed:** đã `confirmed` thì không bao giờ hạ trạng thái. Một miss chỉ đưa score 1 → 5/6, vẫn ≥ 0.6, nên track được giữ.
   - **Xoá** (`should_delete_track`, OR của các điều kiện):
     - (a) `P[0,0] > max_P` hoặc `P[1,1] > max_P` (= 3² = 9 m²): bất định vị trí quá lớn, xoá bất kể score;
     - (b) track `confirmed` có `score < 0.6`, tức khoảng 3 miss liên tiếp từ score 1;
     - (c) track chưa confirmed có `score ≤ 0`. Ví dụ một FP lẻ được tạo với score 1/6 và bị xoá ngay ở lượt miss kế tiếp.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

Mọi số liệu bonus được sinh bằng script trong `student/bonus/`:
- `cache_frames.py` chạy detector Part A–C một lần và lưu cache vào `data/`, không commit.
- `bonus_analysis.py` replay tracker (code E–H + `Sensor`/`Filter`/`TrackManager` của platform) với đúng chuỗi nhiễu camera `default_rng(0)` của `fusion-run-lab`. Script `assert` rằng replay không lệch tái tạo **chính xác** `metrics_lidar.json` và `metrics_fused.json` (matches, ghost, miss, `sum_sq_err`).
- Lần chạy chấm điểm trong `student/artifacts/` **không** bị ghi đè: bonus không gọi `fusion-run-lab`.

- **(+4) Phân tích calibration.** Làm lệch extrinsic camera FRONT mà tracker *tin* là đúng (`sens_to_veh ← sens_to_veh · Δ`). Đo camera vẫn đến từ camera thật. Sweep 8 mức: yaw 0.25/0.5/1/2/5° quanh trục up của camera, và tịnh tiến ngang 0.2/0.5/1.0 m. Bảng đầy đủ: `bonus/calibration_results.md` (+ `.csv`); hình: `bonus/bonus_calibration_sweep.png`, `bonus/bonus_innovation_hist.png`.

  | Mức lệch | Cam updates | mean γu (px) | mean NIS | median min d² của track | % track không có đo trong cổng | RMSE fused (m) |
  |---|---|---|---|---|---|---|
  | 0 (baseline) | 509 | −11.9 | 2.25 | 1.64 | 1.2% | 0.1359 |
  | yaw 0.25° | 486 | −20.0 | 4.55 | 4.39 | 6.1% | 0.1732 |
  | yaw 0.5° | 322 | −26.9 | 7.28 | 9.12 | 37.9% | **0.2112** |
  | yaw 1° | 20 | −75.0 | 3.51 | 23.41 | 97.4% | 0.1726 |
  | yaw 2° | 19 | −80.0 | 3.92 | 66.60 | 97.6% | 0.1760 |
  | yaw 5° | 28 | −51.9 | 2.04 | 80.68 | 97.0% | 0.1503 |
  | ngang 0.2 m | 366 | −23.1 | 5.93 | 7.68 | 29.7% | 0.1903 |
  | ngang 0.5 m | 24 | −68.0 | 3.99 | 27.19 | 96.6% | 0.1621 |
  | ngang 1.0 m | 20 | −82.3 | 3.70 | 76.74 | 97.4% | 0.1796 |

  (Chỉ LiDAR: RMSE 0.1503 m. Mọi mức đều giữ 502 matches, 0 ghost, 239 miss, vì camera không đổi vòng đời track.)

  **Triệu chứng trên innovation:**
  - Innovation ngang `γu` lệch có hệ thống, cùng dấu trên mọi track. Histogram dịch hẳn sang trái khi yaw tăng.
  - NIS trung bình tăng từ 2.25 (≈ `dim = 2`, filter nhất quán) lên 7.28 ở 0.5°.
  - `d²` nhỏ nhất của mỗi track tăng đơn điệu: 1.6 → 4.4 → 9.1 → 23 → 67 → 81.

  **Vì sao gating chặn hoặc không chặn:**
  - **Lệch nhỏ (≤ 0.5° hoặc 0.2 m) không bị chặn.** Bias khoảng 10–25 px vẫn nhỏ so với `√S`, vì `S = HPHᵀ + R` với `R = 5² px²` cộng bất định track. `d²` vẫn < 10.60 (χ²₂ 99.5%), nên đo lệch được chấp nhận và kéo track lệch ngang. Đây là vùng **nguy hiểm nhất**: RMSE xấu nhất 0.211 m ở 0.5°, tệ hơn cả không dùng camera (0.150 m).
  - **Lệch lớn (≥ 1° hoặc 0.5 m) bị chặn gần như hết.** Khoảng 97% track không có đo nào lọt cổng, số camera update rơi từ 509 xuống khoảng 20, và RMSE quay về gần mức chỉ LiDAR. Gating hoạt động như cơ chế an toàn, nhưng cái giá là mất hoàn toàn lợi ích của camera.
  - Khoảng 20 update còn lọt ở mức lệch lớn có γu ≈ −75 px. Chúng thuộc các **track mới sinh**: `P` vận tốc ban đầu lớn (σ = 50 m/s) làm `S` lớn, nên cổng rộng ra và cho đo lệch đi qua.
  - Không đơn điệu: RMSE ở 5° (0.150) thấp hơn ở 1° (0.173), vì ở 5° gần như không còn update nào ảnh hưởng tới track đã hội tụ.

  Kết luận: gating χ² chỉ phát hiện calibration lệch *lớn*. Lệch *nhỏ* cần theo dõi bias của innovation (mean γ ≠ 0, NIS > dim) theo thời gian.

  **Bias khi calibration đúng:** γu ≈ −12 px, γv ≈ +8 px. Nguyên nhân là tâm hộp 2D GT không phải hình chiếu của tâm hộp 3D (xe nhìn xiên, phần thân bên làm hộp 2D lệch). Đây là giới hạn của đo camera mô phỏng từ nhãn.

- **(+3) Trực quan hoá tác dụng của camera update** (4 hình):
  - `bonus/bonus_camera_update_overlay.png`: frame 88, track 0, ảnh FRONT.
    - Đỏ ×: `h(x)` sau update LiDAR, `u = 537.2` px.
    - Xanh lá +: đo camera, `u = 505.6` px.
    - Xanh dương ◆: `h(x)` sau update camera, `u = 511.7` px.

    EKF kéo hình chiếu 25.5 px về phía đo; trong 3D, trạng thái dịch Δy = +0.24 m, Δx = −0.06 m. Camera chủ yếu sửa vị trí **ngang**, vì không đo độ sâu.
  - `bonus/bonus_track_camera_effect.png`: cùng xe GT trong hai mode (track 0 trong cả hai) qua 195 frame. Panel dưới là độ dịch trạng thái do camera update ở cả 199 frame: Δy khoảng 0.05–0.24 m, luôn cùng dấu, đúng với bias γu ở trên.
  - `bonus/bonus_per_vehicle_rmse.png`: RMSE theo từng xe (≥ 20 frame).
    - Xe `mV--0d`: 0.122 → 0.092 m.
    - Xe `8EFRSw`: 0.195 → 0.140 m.
    - Xe `jT38g8`: **xấu đi** 0.148 → 0.166 m, vì bias tâm 2D/3D ở trên.

    Tổng thể fused vẫn tốt hơn (0.150 → 0.136 m). Cải thiện trung bình không có nghĩa là mọi track đều được cải thiện.
  - `bonus/bonus_innovation_hist.png`: histogram γu ở 0°, 1°, 2°.

- **(+3) Export CVAT.**
  - `fusion_lab.export_cvat.export_tracks_json` → `bonus/cvat_tracks_fused.json` (lần chạy chấm điểm) và `bonus/cvat_tracks_fused_109636.json`.
  - XML CVAT for video 1.1 (box = hình chiếu hộp 3D của track lên camera FRONT, thuộc tính `track_id`): `bonus/cvat_annotations_109636.xml`.
  - Segment mặc định **không có** ghost hay đổi ID (`bonus/cvat_id_events.json`), nên dùng segment khoá học `10963653239323173269_1924_000_1944_000` (cùng code, seed 0). Tìm tự động (`bonus/cvat_id_events_109636.json`):
    - **2 lần đổi ID** trên cùng chiếc xe bạc phía trước (~40–44 m): track 16 (frame 57–67) → 18 (frame 73–76) → 20 (từ frame 91). Kiểm tra trong cache detection:
      - Detector bỏ sót xe ở frame 66–68 (3 miss liền). Score 1 → 0.5 < `delete_threshold = 0.6`, nên track 16 bị xoá. Track mới sinh ở frame 69 và cần 5 hit để confirmed (frame 73, ID 18).
      - Detector bỏ sót tiếp ở frame 75–78, nên track 18 bị xoá. Track sinh lại ở frame 79, nhưng còn miss xen kẽ ở các frame 83, 85, 87, 89, nên đến frame 91 mới confirmed (ID 20).
      - Đây là đánh đổi của tham số vòng đời: xoá nhanh thì ít ghost, nhưng bị phân mảnh ID khi detector chập chờn ở xa.
    - **Ghost** frame 136–138 (track 20, x ≈ 50 m): xe vẫn có thật, nhưng tâm GT đã vượt mép cửa sổ đánh giá `lim_x = 50 m` trong khi track ước lượng còn ở 49.9 m. Đây là ghost do biên cửa sổ, không phải FP.
  - Preview vẽ lại từ XML: `bonus/cvat_preview_109636.png`. Hướng dẫn import và chụp ảnh: `bonus/CVAT_HUONG_DAN.md`.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude (Claude Code, model Claude Opus)
- Dùng cho phần nào (hàm, câu hỏi, debug): Claude viết code Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`) theo gợi ý `# vi: TODO` và bộ test; viết script bonus (`student/bonus/cache_frames.py`, `bonus_analysis.py`); soạn nháp phần giải thích và phân tích số liệu trong file này; viết job SLURM để chạy `fusion-run-lab`. Một lỗi đã được debug cùng AI: numpy 2.x không cho `float()` trên ma trận 1×1 trong `mahalanobis_distance`, đã sửa bằng `.item()`.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức):
  - `pytest student/tests -q`: 128 passed, 0 failed/xfailed.
  - Đối chiếu công thức EKF/pinhole/χ² với `docs/HUONG_DAN_KY_THUAT.md` và giáo trình (Thrun et al., ch. 3).
  - Chạy `fusion-run-lab --fusion compare --seed 0` trên frame 0–198. Mọi số liệu trong báo cáo lấy trực tiếp từ `metrics.json` và `grade_run*.log` của lần chạy này, không sửa tay.
  - Script bonus tự `assert` rằng replay không lệch calibration tái tạo đúng `metrics_lidar.json` và `metrics_fused.json`.
  - Tôi đã đọc lại từng hàm và có thể giải thích khi vấn đáp.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa) — không sửa
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))

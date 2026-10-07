# Báo cáo Day 6: Đánh giá độ nhạy và kiểm thử Calibration Drift cho hệ thống LiDAR-Camera Fusion

- **Họ tên:** Cao Đức Hiếu
- **MSSV:** 2A202602701
- **Lớp:** AI20K Track 4
- **Link repo:** https://github.com/ChunHieu/CaoDucHieu-2A202602701-Track4-Day21
- **Topic:** A — LiDAR-camera projection QA
- **Dataset:** data/kitti_mini, data/synthetic, data/nuscenes_mini_subset
- **Các frame đã dùng:** 000011, 000021, 000010, 000049, 000007, scene-0103_010, 000000

---

## 1. Claim

Độ lệch calibration góc xoay quanh trục thẳng đứng (Yaw drift) từ $1.0^\circ$ trở lên gây dịch chuyển hình chiếu trung bình $15.41\text{ px}$, làm sụt giảm trên $41.6\%$ lượng điểm LiDAR rơi đúng vào 2D bounding box của các vật thể ở cự ly xa ($> 30\text{ m}$), trong khi vật thể ở cự ly gần ($< 15\text{ m}$) chỉ mất $10.9\%$ điểm; sự bất đối xứng về khoảng cách này có thể được giám sát tự động theo thời gian thực bằng chỉ số căn chỉnh cạnh (Edge Alignment Score - EAS) với ngưỡng cảnh báo $\text{EAS} < 0.635$.

---

## 2. Evidence

Thực nghiệm sweep được thực hiện trên 5 frame đa dạng của dataset KITTI (`000011`, `000021`, `000010`, `000049`, `000007`) với 39 cấu hình biến thiên extrinsic (Yaw, Pitch, Roll và độ dịch chuyển tịnh tiến $t_x, t_y, t_z$). Số liệu chi tiết được lưu trong file `results/calibration_drift_benchmark.csv`.

### Bảng số liệu benchmark tổng hợp theo mức Perturbation

| Cấu hình / Mức perturb | Pixel Shift TB (px) | Retention Near (<15m) | Retention Far (>30m) | Retention Tổng thể | EAS Score | Ghi chú |
|---|---|---|---|---|---|---|
| Baseline (Không lệch) | 0.00 px | 100.0% | 100.0% | 100.0% | 0.6571 | Calibration chuẩn |
| Yaw +0.5° | 7.73 px | 94.45% | 79.31% | 89.87% | 0.6449 | Vật xa mất 20.7% điểm |
| Yaw +1.0° | 15.41 px | 89.11% | 58.33% | 80.33% | 0.6352 | Vật xa mất 41.7% điểm |
| Yaw +1.5° | 23.07 px | 83.87% | 43.08% | 70.44% | 0.6269 | Rơi vào vùng nguy hiểm |
| Yaw +2.0° | 30.69 px | 78.49% | 33.58% | 62.20% | 0.6232 | Vật xa mất 66.4% điểm |
| Pitch +1.0° | 13.15 px | 91.61% | 53.13% | 78.85% | 0.5949 | Lệch cao độ, EAS tụt mạnh |
| Pitch +1.5° | 19.71 px | 87.32% | 37.68% | 70.11% | 0.5667 | Mất điểm trên trần xe |
| Roll +1.0° | 5.26 px | 95.68% | 86.50% | 94.10% | 0.6597 | Ít nhạy hơn Yaw/Pitch |
| Translation X +10 cm | 2.74 px | 98.16% | 99.01% | 98.32% | 0.6561 | Lệch ngang rất nhỏ ở cự ly xa |
| Translation Y +10 cm | 6.18 px | 95.83% | 93.00% | 94.72% | 0.6500 | Lệch cao độ tịnh tiến |
| Translation Z +10 cm | 6.20 px | 97.01% | 97.74% | 97.03% | 0.6591 | Lệch dọc trục xe |

![Biểu đồ tổng hợp](../results/figures/drift_benchmark_curves.png)

![Demo Overlay chuẩn](../results/figures/demo_overlay_kitti_000011.png)

---

## 3. Failure case

Trong quá trình stress test và chiếu điểm LiDAR lên ảnh 2D, hai failure case nghiêm trọng đã được phát hiện và ghi nhận bằng hình ảnh:

1. **Failure Case 1 — Parallax Bleed & Misalignment (Hình `fail_01_occlusion_parallax_bleed.png`):**
   - **Hiện tượng:** Khi góc Yaw lệch $+1.5^\circ$, các điểm phản xạ từ thân xe ở cự ly trung bình ($z = 26.6\text{ m}$) bị văng ra khỏi cạnh viền 2D box và rơi xuống mặt đường phía sau xe. Đồng thời, các điểm thuộc mặt đường phía sau xe lại chiếu lọt vào bên trong bounding box của chiếc xe.
   - **Lớp lỗi debug:** **Geometry (Hình học)** và **Preprocess (Tiền xử lý)**.
   - **Nguyên nhân gốc:** Thuật toán chiếu 2D trực tiếp không có cơ chế phân loại chiều sâu hoặc loại bỏ điểm che khuất (Occlusion Culling / Z-buffering). Phép chiếu thuần túy hình học biến không gian 3D thành mặt phẳng 2D làm mất thông tin thứ tự trước-sau, dẫn đến việc gán nhầm điểm nền (background road) cho vật thể phía trước (foreground vehicle).

2. **Failure Case 2 — Far Range Point Cloud Sparsity (Hình `fail_02_far_distance_sparsity.png`):**
   - **Hiện tượng:** Với người đi bộ ở cự ly xa ($z = 34.08\text{ m}$), chùm tia LiDAR chỉ bắn trúng 8 điểm. Khi lệch Yaw $1.0^\circ$, toàn bộ các điểm này trượt khỏi bounding box (retention chỉ còn $12.5\%$, và tụt về $0\%$ ở $1.5^\circ$), làm module sensor fusion mất hoàn toàn khả năng liên kết mục tiêu.
   - **Lớp lỗi debug:** **Metric (Cách đo)** và **Geometry (Hình học)**.
   - **Nguyên nhân gốc:** Mật độ điểm LiDAR tỉ lệ nghịch với bình phương khoảng cách ($1/z^2$), trong khi kích thước pixel của 2D box cũng thu nhỏ theo $1/z$. Độ nhạy của các vật thể ở xa với sai số góc là cực kỳ cao, khiến metric in-box retention bị biến thiên gián đoạn (nhảy bước từ có sang mất trắng).

![Failure Case 1](../results/figures/fail_01_occlusion_parallax_bleed.png)

---

## 4. Khuyến nghị nếu triển khai thật

- **Use-case mục tiêu:** Hệ thống hỗ trợ lái xe tự hành ADAS Level 2+/Level 3 và Robot tuần tra ngoài trời.
- **Trade-off cốt lõi:**
  - *Độ chính xác vs Độ trễ tính toán:* Việc chạy thuật toán kiểm tra từng điểm point-by-point trên CPU mất khoảng $15\text{ ms/frame}$. Để giữ pipeline fusion chạy ở tần số camera chuẩn ($30\text{ fps} \approx 33\text{ ms}$), không nên tính metric trên toàn bộ 100.000 điểm của point cloud mà chỉ nên lọc lấy tập silhouette điểm biên ($< 3.000$ điểm) hoặc subsample grid $4\times 4$.
  - *Độ nhạy an toàn vs Cảnh báo giả:* Góc lệch xoay (Rotation) nguy hiểm gấp 5 lần so với sai lệch tịnh tiến (Translation). Hệ thống cần ưu tiên bù góc Yaw/Pitch.
- **Chỉ số hệ thống cần ghi log và cảnh báo khi chạy thật:**
  1. `EAS_score`: Nếu giá trị giảm dưới $0.62$ liên tục trong 10 frame liên tiếp $\rightarrow$ kích hoạt cờ `CALIBRATION_WARNING`.
  2. `far_retention_index`: Theo dõi tỉ lệ điểm bám dính của các tracked vehicle ở cự ly $> 25\text{ m}$.
  3. `chassis_shock_event`: Ghi nhận sự kiện gia tốc kế (IMU) khi xe đi qua ổ gà hoặc va quẹt nhẹ để kích hoạt kiểm tra calibration trực tuyến (Online Self-Calibration Trigger).

---

## 5. Cách chạy lại

Để tái tạo lại toàn bộ kết quả, số liệu CSV và các biểu đồ từ một repo sạch, chạy các lệnh sau:

```bash
# 1. Kích hoạt môi trường và cài đặt thư viện
pip install -r requirements.txt

# 2. Kiểm tra tính toàn vẹn dữ liệu
python tools/verify_data.py --data-root data/kitti_mini
python tools/verify_data.py --data-root data/nuscenes_mini_subset

# 3. Chạy demo chiếu điểm LiDAR lên ảnh camera
python -m starter.projection --data-root data/synthetic --frame 000000
python -m starter.projection --data-root data/kitti_mini --frame 000011
python -m starter.projection --data-root data/nuscenes_mini_subset --frame scene-0103_010

# 4. Chạy toàn bộ thí nghiệm benchmark và sinh biểu đồ/báo cáo
python -m src.drift_experiment

# 5. Tự động kiểm tra tính hợp lệ trước khi nộp bài
python tools/check_submission.py
```

---

## 6. Khai báo sử dụng AI

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| Google Antigravity IDE (Gemini AI) | Hỗ trợ cấu trúc script thí nghiệm benchmark và tối ưu hóa ma trận biến đổi tọa độ | Tự chạy kiểm thử hàm `velo_to_cam` với điểm mẫu $(10, 0, 0)$ để xác nhận $z_{cam} \approx 10\text{ m}$, kiểm tra ảnh overlay trực quan và đối chiếu các con số trong CSV với lý thuyết quang học |

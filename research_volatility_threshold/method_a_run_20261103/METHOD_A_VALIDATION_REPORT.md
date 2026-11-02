# Method A — kết quả validation BTC/ETH

**Trạng thái: không có β khả thi; chưa chọn adaptive threshold.** Không train lại MLP, không đọc predictions test. Không đánh giá performance V_check và không chạy coverage-matched vì chưa có β thỏa protocol đã khóa. Không thay baseline τ=0,64 hoặc tự nới constraint.

## Cấu hình

Volatility 30 dòng đã lưu, chuẩn hóa bằng median/IQR theo coin chỉ trên V_fit. h = tanh((v−median)/IQR); threshold = clip(τ_fixed + βh, 0,50, 0,95). Dùng chính xác τ_fixed trong selection.json (0.6400000000000001). Grid β từ −0,10 đến +0,10, bước 0,02. Objective: mean net bps/mẫu chọn; phí 1 bps. Coverage tổng và **từng coin** phải ≥10%; ít nhất 200 mẫu chọn/coin. Mọi hệ số dùng chung BTC/ETH.

Ranh giới theo timestamp chung, 50%/25%/25% khoảng lịch validation. Phục hồi future timestamp bằng shift −600 dòng riêng từng symbol trên unified; kiểm tra future price khớp tuyệt đối float32 với predictions và ret_bps khớp tới 1e−10 bps. Purge mẫu có endpoint label vượt ranh giới đoạn, không giả định horizon đúng 600 phút.

## Chia dữ liệu và purge

| slice | symbol | start | end_exclusive | before | retained | purged |
|---|---|---|---|---|---|---|
| fit | BTC_USDT | 2024-06-23 23:34:00+00:00 | 2024-07-21 03:24:00+00:00 | 19611 | 19606 | 5 |
| fit | ETH_USDT | 2024-06-23 23:34:00+00:00 | 2024-07-21 03:24:00+00:00 | 16835 | 16805 | 30 |
| select | BTC_USDT | 2024-07-21 03:24:00+00:00 | 2024-08-03 17:19:00.000000001+00:00 | 9850 | 9341 | 509 |
| select | ETH_USDT | 2024-07-21 03:24:00+00:00 | 2024-08-03 17:19:00.000000001+00:00 | 10243 | 9720 | 523 |
| check | BTC_USDT | 2024-08-03 17:19:00.000000001+00:00 | 2024-08-17 07:14:00.000000001+00:00 | 10748 | 10748 | 0 |
| check | ETH_USDT | 2024-08-03 17:19:00.000000001+00:00 | 2024-08-17 07:14:00.000000001+00:00 | 12973 | 12960 | 13 |

Tổng sau purge: V_fit 36.411, V_select 19.061, V_check 23.708; tổng loại 1.080 mẫu. V_check chỉ được xác định membership theo timestamp, chưa dùng correctness/return để đánh giá hoặc chọn β. Các outcome của toàn validation đã được xem trong Bước 1, nên việc chia lại không tạo holdout độc lập.

## Grid trên V_select

Các cột coverage và accuracy trong bảng là %. Returns là bps. BTC coverage / ETH coverage tính trên mẫu số riêng từng coin.

| beta | selected_n | coverage | selective_accuracy | net_selected_bps | expected_net_bps | BTC_USDT_coverage | ETH_USDT_coverage |
|---|---|---|---|---|---|---|---|
| -0.1 | 3668 | 19.24 | 81.92 | 181.869 | 34.998 | 33.38 | 5.66 |
| -0.08 | 3652 | 19.16 | 81.87 | 182.165 | 34.902 | 33.28 | 5.59 |
| -0.06 | 3634 | 19.07 | 81.81 | 182.584 | 34.81 | 33.15 | 5.52 |
| -0.04 | 3600 | 18.89 | 81.64 | 182.806 | 34.526 | 32.81 | 5.5 |
| -0.02 | 3532 | 18.53 | 81.34 | 183.356 | 33.976 | 32.1 | 5.49 |
| 0.0 | 3462 | 18.16 | 81.02 | 183.929 | 33.406 | 31.36 | 5.48 |
| 0.02 | 3494 | 18.33 | 81.05 | 183.303 | 33.601 | 31.7 | 5.48 |
| 0.04 | 3502 | 18.37 | 80.95 | 182.802 | 33.586 | 31.78 | 5.48 |
| 0.06 | 3534 | 18.54 | 80.62 | 181.165 | 33.589 | 32.11 | 5.5 |
| 0.08 | 3505 | 18.39 | 80.51 | 180.181 | 33.132 | 31.8 | 5.5 |
| 0.1 | 3463 | 18.17 | 80.25 | 178.603 | 32.449 | 31.37 | 5.48 |

**Nguyên nhân dừng:** ETH coverage chỉ 5,48–5,66%, thấp hơn 10% cho mọi β; số mẫu ETH 533–550 vẫn vượt minimum 200. Coverage tổng 18,16–19,24% đều vượt 10%. Nếu chỉ ràng buộc coverage tổng thì các ứng viên sẽ hợp lệ, nhưng đó là một protocol khác. Không thay protocol sau khi thấy kết quả.

Ngay cả khi chỉ nhìn objective và bỏ qua feasibility, β=0 đạt mean net cao nhất 183,929 bps/mẫu chọn; không có β khác cải thiện metric này trên V_select. Tuy nhiên β=−0,10 đạt expected net/mẫu đủ điều kiện 34,998 bps, cao hơn 33,406 ở β=0 do chọn thêm mẫu; đây là objective khác, không dùng để đổi nghiệm sau thử nghiệm. Không thể kết luận adaptive thất bại ở mọi hoàn cảnh hoặc trên test.

## Giới hạn và bước tiếp theo

Cohort oracle-filtered |future return|>1%; kết quả không phải backtest có thể giao dịch. Labels 600 dòng chồng lấn; không coi selections là số vị thế độc lập. Calibration/model selection và fixed threshold cũ dùng validation; toàn bộ nghiên cứu vẫn exploratory. Không có significance IID hoặc annualized Sharpe.

Giữ fixed 0,64. Nếu tiếp tục nghiên cứu, cần chốt trước liệu minimum coverage áp dụng pooled hay từng coin, rồi mở một thí nghiệm mới có nhãn rõ; không thay kết quả lần này. Hiện tại **NO-GO cho việc chuyển sang adaptive đã chọn**, không phải kết luận NO-GO cho mọi volatility-adaptive method.

## Artifacts và tái lập

- Script: `research_volatility_threshold/method_a.py`.
- Chạy lại cùng protocol vào thư mục mới: `reproduction_setup/venv/Scripts/python.exe research_volatility_threshold/method_a.py --out research_volatility_threshold/method_a_reproduction_NEW`.
- `beta_grid_select.csv`: đủ 11 ứng viên, mọi feasibility=False.
- `split_counts.csv`: timestamps, counts và purge.
- `selection_grid.png`: coverage và objective theo β.
- `resources.jsonl`: các lần kiểm tra working set, peak, private commit và available RAM; guard 9 GiB working set, dừng nếu available <2 GiB. Đây là kiểm tra theo stage, không phải continuous sampler.
- `status.json`: infeasible. Không có model mới hoặc selected adaptive configuration.

Peak working set ghi nhận: 0.376 GiB; available RAM thấp nhất được ghi nhận: 9.242 GiB. Runtime ghi nhận: 8.08 giây (không gồm bước hoàn thiện báo cáo/biểu đồ).

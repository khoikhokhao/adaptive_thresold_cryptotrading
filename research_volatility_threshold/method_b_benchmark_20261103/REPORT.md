# Fixed 0,64, Method A và Method B — cùng validation/test BTC/ETH

**Kết luận:** B tăng net/mẫu chọn trên test nhưng không thắng Fixed tổng thể: accuracy chỉ tăng khoảng 0,21 điểm phần trăm, coverage giảm và expected net/mẫu đủ điều kiện giảm. BTC vẫn có net âm và xấu hơn Fixed; ETH chọn chính xác hơn. Chưa có bằng chứng cải thiện bền vững hoặc có ý nghĩa thống kê. Giữ kết quả Fixed/A cũ và chưa thay baseline chính bằng B.

## Protocol

Giữ nguyên MLP, probabilities gốc, labels, phí 1 bps và cohort baseline: validation 80.260, test 65.182 mẫu. Lưu nguyên toàn bộ kết quả A ở thư mục full_validation_test_20261103; hash mọi artifact trước và sau chạy, kiểm tra metrics A/Fixed khớp kết quả cũ (PASS).

Fit B trên toàn validation, chọn κ trên toàn validation theo mean net bps/mẫu chọn cao nhất với coverage pooled ≥10%; không ràng buộc từng coin hoặc test. Median/IQR volatility tái sử dụng từ A, không fit trên test. Model meta-selection B khác với MLP: chỉ fit 4 hệ số correctness, không train lại MLP hoặc thay hướng dự đoán.

## Công thức và cấu hình

`h = tanh((volatility_30m − median_symbol)/IQR_symbol)`; `l = logit(clip(confidence, 1e−6, 1−1e−6))`.

`q_correct = sigmoid((1 + delta*h)*l + a_symbol + b*h)`; chọn khi `q_correct >= kappa`. Đây là xác suất correctness có điều kiện, không phải probability Up mới. |delta|≤0,25 bảo toàn tính đơn điệu theo confidence. Loss: binary cross-entropy correctness, tổng trọng số mỗi coin 0,5, cộng 0,01 × tổng bình phương hệ số; λ không tune. L-BFGS-B với gradient giải tích đã đối chiếu finite differences trên dữ liệu thật. Khởi tạo zero, không retry cấu hình khác. Grid κ=0,50–0,94 bước 0,02; chọn trên validation. Hệ số, κ và toàn bộ cutoff coverage-matched được lưu trước khi đọc test; không refit sau test.

Đối chứng `b_symbol_only`: cùng fit/selection nhưng b=delta=0, còn hai intercept asset. Đây là ablation nhằm tách hiệu ứng volatility khỏi hiệu chỉnh theo symbol. Method A beta=−0,06 và fixed τ=0,64 giữ nguyên.

## Hệ số và trạng thái fit

{
  "models": {
    "method_b": {
      "coefficients": [
        -0.04651906715011452,
        -0.013321429975365949,
        0.12373567958046972,
        0.10809409948849082
      ],
      "loss": 0.6671086770342891,
      "iterations": 9,
      "success": true,
      "message": "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
      "full": true,
      "gradient_check": "PASS",
      "delta_at_bound": false
    },
    "b_symbol_only": {
      "coefficients": [
        -0.02250784469727953,
        0.022425501937632063
      ],
      "loss": 0.6683290045967601,
      "iterations": 3,
      "success": true,
      "message": "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
      "full": false,
      "gradient_check": "PASS",
      "delta_at_bound": false
    }
  },
  "kappas": {
    "method_b": 0.68,
    "b_symbol_only": 0.64
  }
}

## Metrics tổng hợp

Coverage và selective accuracy là %. Net là bps. `expected_net_bps = coverage × net_selected_bps`. Các method không có coverage bằng nhau ở operating point native.

| split | method | symbol | selected_n | coverage | selective_accuracy | net_selected_bps | expected_net_bps |
|---|---|---|---|---|---|---|---|
| validation | fixed_064 | pooled | 20463 | 25.5 | 68.93 | 99.181 | 25.287 |
| validation | method_a | pooled | 18899 | 23.55 | 71.93 | 117.122 | 27.579 |
| validation | method_b | pooled | 11836 | 14.75 | 75.68 | 139.432 | 20.562 |
| validation | b_symbol_only | pooled | 20465 | 25.5 | 68.94 | 99.193 | 25.293 |
| test | fixed_064 | pooled | 7984 | 12.25 | 68.75 | 87.128 | 10.672 |
| test | method_a | pooled | 7434 | 11.4 | 68.39 | 89.516 | 10.209 |
| test | method_b | pooled | 6633 | 10.18 | 68.96 | 95.304 | 9.698 |
| test | b_symbol_only | pooled | 8017 | 12.3 | 68.88 | 87.618 | 10.777 |

## Test theo asset

| split | method | symbol | selected_n | coverage | selective_accuracy | net_selected_bps | expected_net_bps |
|---|---|---|---|---|---|---|---|
| test | fixed_064 | BTC_USDT | 3808 | 13.13 | 49.13 | -44.294 | -5.815 |
| test | fixed_064 | ETH_USDT | 4176 | 11.54 | 86.64 | 206.969 | 23.89 |
| test | method_a | BTC_USDT | 3509 | 12.1 | 46.17 | -55.082 | -6.664 |
| test | method_a | ETH_USDT | 3925 | 10.85 | 88.25 | 218.788 | 23.737 |
| test | method_b | BTC_USDT | 3463 | 11.94 | 46.09 | -55.442 | -6.62 |
| test | method_b | ETH_USDT | 3170 | 8.76 | 93.94 | 259.984 | 22.78 |
| test | b_symbol_only | BTC_USDT | 3808 | 13.13 | 49.13 | -44.294 | -5.815 |
| test | b_symbol_only | ETH_USDT | 4209 | 11.63 | 86.74 | 206.962 | 24.078 |

## Coverage-matched và thời gian

coverage_matched_metrics.csv gồm cutoff chọn bằng validation, targets 10%, 20%, 25,5%, 40%, pooled và từng symbol. Fixed score=confidence; A score=confidence−beta*h (diagnostic không clip); B score=q_correct. Không dùng test quantile/top-k. Khi plateau khiến target không khả thi trong ±1 điểm phần trăm, attainable=False và có hai coverage khả thi kề nhau; actual_on_target cho biết drift trên test. Chỉ coi matched nếu cả hai method đều đạt tolerance thực tế trên cùng scope; biểu đồ không chứng minh ranking tốt hơn khi actual coverage khác nhau.

weekly_metrics.csv báo stability theo tuần và asset; mọi số này chỉ để đánh giá sau khi khóa, không chọn lại method hoặc tham số. Mẫu chọn <200/tuần/asset được gắn few_samples. Không kiểm định IID hoặc công bố statistical significance vì labels và returns chồng lấn.

## Diễn giải so sánh

- Validation: B đạt selective accuracy 75,68%, net/mẫu chọn 139,43 bps, coverage 14,75%. A tương ứng 71,93%, 117,12 bps, 23,55%; Fixed 68,93%, 99,18 bps, 25,50%. B chọn ít mẫu hơn đáng kể; không diễn giải toàn bộ mức tăng accuracy là lợi thế volatility.
- Test: B đạt accuracy 68,96%, net/mẫu chọn 95,30 bps, coverage 10,18%; Fixed 68,75%, 87,13 bps, 12,25%; A 68,39%, 89,52 bps, 11,41%.
- Expected net/mẫu đủ điều kiện test: Fixed **10,67**, A **10,21**, B **9,70** bps. Theo metric này Fixed tốt hơn cả hai adaptive tại operating points được chọn trên validation.
- B trên test BTC: accuracy 46,09%, net −55,44 bps/mẫu chọn; ETH 93,94%, +259,98 bps/mẫu chọn. Không dùng ETH để che mất kết quả BTC.
- Đối chứng chỉ có intercept symbol đạt net/mẫu chọn test 87,62 bps, expected net 10,78 bps, coverage 12,30%. B không thắng đối chứng này về expected net. Không chọn đối chứng làm model mới dựa trên test; đây chỉ là ablation đã khóa từ validation.
- Ở điểm diagnostic target 10%, actual test coverage xấp xỉ nhau: Fixed 9,98%, A 9,83%, B 9,96%; net/mẫu chọn lần lượt 94,79; 97,26; 95,37 bps. B không cho thấy thắng A tại điểm này. Fixed không đạt target 10% trên validation do plateau (gần nhất 8,57%), nên đây là so sánh **gần coverage trên test**, không phải matching chính xác đã đạt ở cả hai tập. Không dùng test để ép cutoff.
- Các target validation 20%, 25,5%, 40% có coverage drift lớn trên test. Không so trực tiếp các return ở những điểm có coverage khác nhau như thể đã matched. Xem các cờ attainable và actual_on_target trong CSV.

Hệ số B: a_BTC=−0,046519, a_ETH=−0,013321, b=+0,123736, delta=+0,108094; κ=0,68. κ là ngưỡng trên **q_correct**, không phải fixed threshold confidence MLP=0,68. Confidence threshold hiệu dụng của B trên test có median khoảng 0,6966 BTC và 0,6813 ETH. Fit hội tụ sau 9 iterations, delta không chạm bound; không có dấu hiệu optimizer bị kẹt ở ràng buộc, nhưng hội tụ không chứng minh tổng quát hóa.

Runtime phần tính toán/logging khoảng 3,51 giây, peak working set khoảng 0,238 GiB; không train lại MLP. Toàn bộ 4 hệ số và κ fit/chọn trên full validation, khóa rồi mới đọc test trong script này. Kết quả A và Fixed được kiểm tra khớp số cũ, hash mọi artifact cũ không thay đổi.

## Giới hạn

Oracle filtering |future return|>1%, horizon 600 dòng, overlap, thiếu purge và calibration/threshold cùng dùng validation được giữ nguyên để so sánh đồng nhất baseline. B fit reliability và chọn κ trên cùng validation cũng có nguy cơ overfit. Test đã được xem ở các lần trước, nên mọi kết quả là retrospective/exploratory. Net/mẫu không phải portfolio PnL hay lợi nhuận annualized. Không tuyên bố novelty hoặc confirmatory OOS. Không chọn lại cấu hình theo bảng test.

## Artifacts

Script: research_volatility_threshold/method_b_benchmark.py. Lệnh: `reproduction_setup/venv/Scripts/python.exe research_volatility_threshold/method_b_benchmark.py` (từ chối overwrite thư mục kết quả). Có locked_config_before_test.json, hai kappa grids validation, metrics_validation_test.csv, coverage_matched_metrics.csv, weekly_metrics.csv, macro_average_metrics.csv, threshold_distributions.csv, validation/test_decisions.parquet, risk_profit_coverage.png, previous_results_sha256.json, resources.jsonl và completion.json. Memory guard kiểm tra theo stage: peak working set ≤9 GiB và available ≥2 GiB.

# Regime-B — Direction–Trend Interaction Threshold

Tài liệu: [PDF/context gốc](README.md) · [Analysis](../analysis/REGIME_EXPLORATORY_ANALYSIS.md) · [Source code](../research_regime_threshold/run_regime_b.py).

## 1. Limit của cách làm gốc → Idea & Hypothesis

Một threshold chung chưa thể hiện trực tiếp việc cùng past trend có thể mang reliability/payoff khác nhau cho dự đoán Up và Down. **Hypothesis:** threshold phụ thuộc trend và predicted direction giúp chọn mẫu tốt hơn, thay vì coi confidence có ý nghĩa giống nhau ở mọi regime/direction.

Validation: ETH Bear/Bull net/chọn−93,87/+98,68 bps tại Fixed 0,64; ETH Bull predicted Down accuracy 99,38% trên 324 selections, Up58,00% trên 1.612. BTC Bear/Bull lại khá tương đồng. Đây là heterogeneity, **chưa phải interaction ổn định**, còn chịu volatility/time/asset composition. Nguồn: Analysis ở đầu tài liệu.

## 2. Proposed Method

Theo từng symbol, P là mid-price, r_t=log(P_t/P_(t−1)) không clip. Tính từ full observed history, không từ cohort đã lọc Neutral:

```math
T_t=\frac{\log P_t-\log P_{t-360}}{\sqrt{\sum_{j=0}^{359}r_{t-j}^{2}}},\quad z_t=\tanh(T_t/2),\quad
\tau_t=\mathrm{clip}(0.64+\beta z_t+\gamma u_tz_t+\eta u_t,0.50,0.95).
```

360 là dòng; scale 2 cố định. u=+1/−1 theo **predicted** Up/Down, tuyệt đối không dùng future label. β là trend effect chung; γ cho phép trend slope khác theo direction; η sửa baseline bias Up/Down. Cụ thể trước clip: Upτ=0,64+η+(β+γ)z; Downτ=0,64−η+(β−γ)z.

Chọn nếu c=max(p_up, p_down)≥τ. Chọn+Up→Long, chọn+Down→Short; còn lại Abstain. Missing/zero denominator: Fixed fallback; β=γ=η=0 khôi phục Fixed.

**Học tham số:** θ=(β,γ,η), e là selection, g=uR_bps−1:

```math
\widehat\theta=\arg\max_{\theta}\left[\frac{\sum_{V}e_t(\theta)g_t}{\sum_{V}e_t(\theta)}-100\|\theta\|_2^{2}\right],\quad
\frac{\sum_Ve_t}{N_V}\ge0.10,\quad |\beta|+|\gamma|+|\eta|\le0.16.
```

Mỗi coefficient grid−0,10…+0,10 bước 0,02:719 candidates sau constraint. Chọn trên full validation, penalty đơn vị bps; không tune lại window/scale/λ. Hòa 10⁻⁸ bps: norm nhỏ rồi lexicographic. 3 parameters chung hai coin; không bắt coverage từng coin/direction/test ≥10%.

## 3. Benchmark Plan + Kết quả kỳ vọng

Fixed 0,64; Fixedτ*; B; γ=0; direction-only β=γ=0; alignment-only β=η=0; trend-only γ=η=0. Refit ablations bằng validation theo cùng objective. Metrics n/coverage/selective accuracy/net-chọn/expected-net-eligible; breakdown BTC/ETH, Up/Down và tuần. Matching diagnostic score=c−(βz+γuz+ηu), không clip; cutoff validation đóng băng, theo protocol chung.

| Split | Method | n chọn | Coverage % | Selective accuracy % | Net/chọn, bps | Expected net/eligible, bps |
|---|---|---:|---:|---:|---:|---:|
| Validation | Fixed | 20.463 | 25,4959 | 68,93 | 99,181 | 25,287 |
| Validation | Regime-B | 20.463 | 25,4959 | 77,21 | 113,067 | 28,827 |
| Test lịch sử | Fixed | 7.984 | 12,2488 | 68,75 | 87,128 | 10,672 |
| Test lịch sử | Regime-B | 7.984 | 12,2488 | 77,00 | 99,326 | 12,166 |

So sánh trên cùng số mẫu được chọn với Fixed 0,64; coverage validation 25,4959%, test 12,2488%. Expected net/eligible = coverage × net/chọn; số liệu được tính trước khi làm tròn hiển thị.

Hyperparameters, normalization và quy tắc tối ưu ở mục 2 giữ nguyên; không chọn tham số bằng test.

## 4. Limitations & Expected Contribution

Kỳ vọng: tách trend effect, direction bias và alignment bằng nested ablations. Trong kết quả v1, γ=0; interaction chưa có evidence, nên cần giữ đối chứng bỏ interaction và direction-only. η có thể chỉ đổi allocation; trend có thể đại diện volatility hoặc một episode. Novelty/generalization chưa xác minh; oracle filtering, overlap, validation reuse và test đã xem vẫn tồn tại.


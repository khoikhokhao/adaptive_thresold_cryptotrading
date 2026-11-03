# ACF-A — Shrunk Absolute-ACF Tilt

Tài liệu: [PDF/context gốc](README.md) · [Analysis](../analysis/AUTOCORRELATION_EXPLORATORY_ANALYSIS.md) · [Source code](../research_autocorrelation_threshold/run_acf_a.py).

## 1. Limit của cách làm gốc → Idea & Hypothesis

Fixed τ chưa trực tiếp điều chỉnh selection theo độ phụ thuộc thời gian của returns. Không khẳng định MLP không biết temporal information qua features. **Hypothesis:** absolute ACF mang reliability information còn dư ở cùng confidence; một tilt một hệ số đủ kiểm chứng mà không thêm predictive model mới.

Validation |ACF lag-1, W=360| Q1→Q5: selective accuracy BTC 62,46→83,55%, ETH 47,94→72,96%; net/chọnBTC 106,43→207,67 bps, ETH−118,07→192,47. Nhưng sau confidence/direction/volatility/trend/day controls, correctness effect/IQR là BTC−0,85pp, ETH+3,56pp. Raw association không phải incremental value ổn định. Nguồn: Analysis ở đầu tài liệu.

## 2. Proposed Method

Theo từng coin, r_t=log(P_t/P_(t−1)) từ mid-price quan sát được; W=360, k=1. Dùng common window mean, không phải Pearson với hai mean riêng:

```math
\bar r_t=\frac{1}{W}\sum_{j=0}^{W-1}r_{t-j},\quad
\rho_t(k)=\frac{\sum_{j=0}^{W-k-1}(r_{t-j}-\bar r_t)(r_{t-j-k}-\bar r_t)}{\sum_{j=0}^{W-1}(r_{t-j}-\bar r_t)^{2}},\quad a_t=|\rho_t(1)|.
```

Factor dùng full past unified history, không clip log returns hoặc rolling trên oracle-filtered cohort. Window theo dòng, không bảo đảm phút.

```math
h_t=\tanh\frac{a_t-m_s}{I_s},\quad \tau_t=\mathrm{clip}(0.64+\beta h_t,0.50,0.95).
```

m_s/I_s là median/IQR **full validation** từng coin, đóng băng sang test; temporal diagnostics phải fit lại trên past prefix. β<0 hạ ngưỡng khi |ACF| cao; β>0 nâng ngưỡng; β=0 khôi phục Fixed. Không dùng dấu ACF để đổi hướng MLP.

Chọn e=1{c≥τ}, c=max(p_up, p_down). Chọn+Up→Long, chọn+Down→Short; còn lại Abstain. Missing/nonfinite, ACF denominator0 hoặc IQR0: Fixed fallback, không bỏ mẫu.

**Học tham số:** g=uR_bps−1, u là predicted direction±1:

```math
\widehat\beta=\arg\max_{\beta\in\{-0.10,-0.08,\ldots,0.10\}}\left[\frac{\sum_Ve_t(\beta)g_t}{\sum_Ve_t(\beta)}-100\beta^{2}\right],\quad \frac{\sum_Ve_t}{N_V}\ge0.10.
```

1 supervised coefficient + 4 normalization stats; full validation selection; penalty đơn vị bps. Hòa 10⁻⁸ bps ưu tiên |β| nhỏ rồi β nhỏ. Không tune thêm lag/window/λ; không constraint test/từng coin. Infeasible phải báo, không nới coverage.

## 3. Benchmark Plan + Kết quả kỳ vọng

Fixed 0,64; Fixedτ*; ACF-A; nullβ=0. Metrics n, coverage, selective accuracy, net/chọn, expected net/eligible; breakdown coin/direction/tuần. Matching score=c−βh không clip; cutoffs validation, không test top-k. **Within-asset matching và asset-mix decomposition bắt buộc** để phân biệt selection gain với allocation gain; xem protocol chung.

| Split | Method | n chọn | Coverage % | Selective accuracy % | Net/chọn, bps | Expected net/eligible, bps |
|---|---|---:|---:|---:|---:|---:|
| Validation | Fixed | 20.463 | 25,4959 | 68,93 | 99,181 | 25,287 |
| Validation | ACF-A | 20.463 | 25,4959 | 78,59 | 115,050 | 29,333 |
| Test lịch sử | Fixed | 7.984 | 12,2488 | 68,75 | 87,128 | 10,672 |
| Test lịch sử | ACF-A | 7.984 | 12,2488 | 78,37 | 101,069 | 12,380 |

So sánh trên cùng số mẫu được chọn với Fixed 0,64; coverage validation 25,4959%, test 12,2488%. Expected net/eligible = coverage × net/chọn; số liệu được tính trước khi làm tròn hiển thị.

Hyperparameters, normalization và quy tắc tối ưu ở mục 2 giữ nguyên; không chọn tham số bằng test.

## 4. Limitations & Expected Contribution

Kỳ vọng: dependence-aware selection tối giản với exact fixed null và attribution rõ. Absolute ACF bỏ dấu momentum/mean reversion; một β chung có thể sai dấu một coin. Cần tách pooled gain do asset mix khỏi within-asset selection; không thêm lags/VR/ER hậu nghiệm để cứu method. Novelty chưa xác minh; oracle filtering, overlap, validation reuse và test đã xem vẫn tồn tại.


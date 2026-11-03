# Volatility-B — Conditional Correctness Odds

Tài liệu: [PDF/context gốc](README.md) · [Analysis](../analysis/VOLATILITY_VALIDATION_ANALYSIS.md) · [Source code](../research_volatility_threshold/method_b_benchmark.py).

## 1. Limit của cách làm gốc → Idea & Hypothesis

Paper tối ưu một τ chung trên validation rồi giữ cố định (§3.4); calibration chung chưa bảo đảm cùng confidence có correctness như nhau ở mọi volatility state. **Hypothesis:** volatility sửa cả mức reliability nền và mức độ tin cậy của confidence, nên một correction nhỏ có thể chọn mẫu tốt hơn.

Validation: volatility30 dòng Q1→Q5 có selective accuracy BTC 53,74→84,10%, ETH 34,79→67,41%. Tuy nhiên partial correlation với correctness đảo dấu khi kiểm soát ngày: BTC+0,056→−0,101, ETH+0,071→−0,064. Evidence hỗ trợ **kiểm tra**, không chứng minh “volatility cao luôn tốt”. Nguồn: Analysis ở đầu tài liệu.

## 2. Proposed Method

Theo từng coin s, P là mid-price; r̃_j=clip(P_j/P_{j−1}−1,−0,12,+0,12). v_t là sample std(ddof1) của r̃_(t−30)…r̃_(t−1), min15 quan sát. Dùng đúng `volatility_30m` gốc; 30 là dòng, không bảo đảm phút.

```math
h_t=\tanh\frac{v_t-m_s}{I_s},\quad
\ell_t=\operatorname{logit}(\operatorname{clip}(c_t,10^{-6},1-10^{-6})),\quad
q_t=\sigma\left[(1+\delta h_t)\ell_t+a_s+b h_t\right].
```

m_s/I_s là median/IQR **early-validation V_fit đã purge**, tái sử dụng artifact đã khóa; c=max(p_up, p_down); σ là sigmoid. q ước lượng probability **prediction đúng**, không thay probability Up hoặc hướng MLP. a_s sửa reliability riêng coin; b là volatility effect; δ là confidence×volatility interaction.

Chọn khi q≥κ. Trên miền confidence không chạm clipping, threshold tương đương:

```math
\tau_t=\max\left\{0.5,\sigma\left(\frac{\operatorname{logit}\kappa-a_s-bh_t}{1+\delta h_t}\right)\right\},\quad |\delta|\le0.25.
```

Cách đọc: cùng reliability requirement κ tạo confidence threshold khác theo volatility/coin. Bound giữ confidence slope dương; ở biên thực thi bằng q≥κ, không clip τ ở 0,95. Chọn+Up→Long, chọn+Down→Short, còn lại Abstain. Missing factor/IQR0: Fixed fallback, giữ cohort.

**Học tham số:** θ=(a_BTC,a_ETH,b,δ), k=indicator đúng; trên full validation V:

```math
\widehat\theta=\arg\min_{|\delta|\le0.25}\left\{\frac12\sum_s\frac1{N_s}\sum_{V_s}
[-k_t\log q_t-(1-k_t)\log(1-q_t)]+0.01\|\theta\|_2^{2}\right\}.
```

L-BFGS-B từ zero. Sau đó chọn κ∈{0,50;0,52;…;0,94} tối đa mean net/chọn với **validation pooled coverage≥10%**; hòa ưu tiên gần 0,64 rồi nhỏ hơn. 4 coefficients + κ; normalization vẫn early V_fit dù fit θ/chọn κ dùng full V. Null θ=0, κ=0,64 khôi phục Fixed trên cohort hiện tại.

## 3. Benchmark Plan + Result

Fixed 0,64; Fixedτ*; B; symbol-only b=δ=0; bỏ từng b/δ để tách level/slope effects nếu kiểm chứng tiếp. Report n, coverage, selective accuracy, net/chọn, expected net/eligible; logloss/Brier/ECE của q. Coverage diagnostic score=q; cùng BTC/ETH, direction, tuần và cutoffs validation theo protocol chung.

### Mục tiêu benchmark

| Split | Method | n chọn | Coverage % | Selective accuracy % | Net/chọn, bps | Expected net/eligible, bps |
|---|---|---:|---:|---:|---:|---:|
| Validation | Fixed | 20.463 | 25,4959 | 68,93 | 99,181 | 25,287 |
| Validation | Volatility-B | 20.463 | 25,4959 | 75,83 | 111,083 | 28,322 |
| Test lịch sử | Fixed | 7.984 | 12,2488 | 68,75 | 87,128 | 10,672 |
| Test lịch sử | Volatility-B | 7.984 | 12,2488 | 75,63 | 97,584 | 11,953 |

So sánh trên cùng số mẫu được chọn với Fixed 0,64; coverage validation 25,4959%, test 12,2488%. Expected net/eligible = coverage × net/chọn; số liệu được tính trước khi làm tròn hiển thị.

Hyperparameters, normalization và quy tắc tối ưu ở mục 2 giữ nguyên; không chọn tham số bằng test.

## 4. Limitations & Expected Contribution

Kỳ vọng: conditional-reliability threshold có dạng đảo giải tích, tách volatility khỏi symbol calibration. Correctness không phản ánh độ lớn lãi/lỗ; b/δ có thể học episode; fit reliability và chọn κ cùng V dễ overfit. Cần đối chiếu symbol-only ở comparable coverage và kiểm tra độ ổn định theo thời gian. Novelty chưa xác minh; oracle filtering, overlap, validation reuse và test đã xem vẫn tồn tại.


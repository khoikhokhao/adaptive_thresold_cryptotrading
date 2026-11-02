# Shift-A — Shrunk PCA-Residual Threshold

Tài liệu: [PDF/context gốc](README.md) · [Analysis](../analysis/DISTRIBUTION_SHIFT_EXPLORATORY_ANALYSIS.md) · [Source code](../research_distribution_shift_threshold/run_shift_a.py).

## 1. Limit của cách làm gốc → Idea & Hypothesis

Fixed τ và isotonic calibration không trực tiếp conditioning selection vào khoảng cách giữa feature vector hiện tại và training distribution. **Hypothesis:** training PCA residual bổ sung một phần reliability information ngoài confidence; threshold tilt nhỏ có thể khai thác hoặc bác bỏ nó bằng β=0. Không mặc định shift cao là prediction sai.

Exploration: RE32 thêm vào controls confidence/symbol/volatility/trend/ACF/direction cho weighted logloss gain **+0,004530 (~0,54%)**, dương cả 3 folds. Nhưng augmented comparator vẫn kém confidence reference cả 3 folds; return-RMSE gain chỉ 0,110 bps. Đây là incremental proper-score evidence, **không phải calibration success hoặc trading profit**. Nguồn: Analysis ở đầu tài liệu.

## 2. Proposed Method

x là 128 features đã impute/RobustScale của MLP. Theo coin s, reuse training-only mean μ_s, diagonal std D_s và covariance eigenvectors u_(s,j), eigenvalues giảm dần:

$$
z_t=D_s^{-1}(x_t-\mu_s),\quad d_t=\sqrt{\frac1{96}\sum_{j=33}^{128}(u_{s,j}^{\top}z_t)^2},\quad
h_t=\tanh\frac{d_t-m_s}{v_s-m_s},\quad \tau_t=\operatorname{clip}(0.64+\beta h_t,0.50,0.95).
$$

d là RMS spectral residual ngoài 32 PCs, không phải direction; m_s/v_s là **training residual Q50/Q95**. Reference dùng toàn bộ 292.338 train rows, không labels; không refit bằng validation/test. Extra geometry chỉ phục vụ factor, không đổi inputs của MLP. Giữ spectral-tail calculation đã kiểm chứng.

β>0 tăng threshold khi residual cao; β<0 giảm; β=0 khôi phục Fixed. Cho phép cả hai dấu vì correctness và payoff association có thể khác. Chọn e=1{c≥τ}, c=max(p_up, p_down); chọn+Up→Long, chọn+Down→Short, còn lại Abstain. Missing/nonfinite hoặc v≤m: Fixed fallback, giữ mẫu số.

**Học tham số:** g=uR_bps−1, u là predicted direction±1:

$$
\widehat\beta=\arg\max_{\beta\in\{-0.10,-0.08,\ldots,0.10\}}\left[\frac{\sum_Ve_t(\beta)g_t}{\sum_Ve_t(\beta)}-100\beta^2\right],\quad \frac{\sum_Ve_t}{N_V}\ge0.10.
$$

1 supervised coefficient, ngoài unsupervised geometry/stats. Full validation selection; λ 100 bps, grid bước 0,02; hòa10⁻⁸bps ưu tiên |β| nhỏ rồi β nhỏ. PCA32, normalization và λ giữ nguyên. Không constraint từng coin/test; infeasible phải báo, không thay dimension để cứu utility.

## 3. Benchmark Plan + Result

Fixed 0,64; Fixedτ*; Shift-A; nullβ=0. Metrics n, coverage, selective accuracy, net/chọn, expected net/eligible; coin/direction/tuần. Matching diagnostic score=c−βh không clip, cutoffs validation đóng băng. Kiểm tra asset mix, within-asset×direction và conditional validation volatility/trend/ACF strata theo protocol chung.

| Split | Method | n chọn | Coverage % | Selective accuracy % | Net/chọn, bps | Expected net/eligible, bps |
|---|---|---:|---:|---:|---:|---:|
| Validation | Fixed | 20.463 | 25,4959 | 68,93 | 99,181 | 25,287 |
| Validation | Shift-A | 20.463 | 25,4959 | 79,96 | 117,034 | 29,839 |
| Test lịch sử | Fixed | 7.984 | 12,2488 | 68,75 | 87,128 | 10,672 |
| Test lịch sử | Shift-A | 7.984 | 12,2488 | 79,75 | 102,811 | 12,593 |

So sánh trên cùng số mẫu được chọn với Fixed 0,64; coverage validation 25,4959%, test 12,2488%. Expected net/eligible = coverage × net/chọn; số liệu được tính trước khi làm tròn hiển thị.

Hyperparameters, normalization và quy tắc tối ưu ở mục 2 giữ nguyên; không chọn tham số bằng test.

### Cách chạy

Cần matrices/metadata/manifests trong `data/btc_eth_1pct/prepared/`, saved predictions, training PCA reference và cached analysis/folds trong `research_distribution_shift/validation_20261103/`, cùng config/metrics ACF tham chiếu.

Dùng môi trường Python đã chuẩn bị, chạy PowerShell từ repository:

```powershell
Set-Location 'D:\crypto-confidence-execution'
$methodRunOut = 'research_runs/shift_a_' + (Get-Date -Format yyyyMMdd_HHmmss_fff)
& .\reproduction_setup\venv\Scripts\python.exe .\research_distribution_shift_threshold\run_shift_a.py --out $methodRunOut
```

`--out` trỏ đến thư mục có timestamp, chưa tồn tại; script từ chối ghi đè thư mục cũ. Lệnh fit/chọn threshold bằng validation rồi đánh giá test theo cấu hình script; không train lại MLP. Guard theo stage giữ peak working set≤9GiB và Available RAM≥2GiB.

Output trong `research_runs/`: cấu hình đã khóa, grids, `metrics_validation_test.csv`, decisions validation/test, coverage/weekly diagnostics và resource logs. Những input/model/reference paths khác vẫn lấy theo source, không đổi theo `--out`.

## 4. Limitations & Expected Contribution

Kỳ vọng: training-reference-aware selection đơn giản và có kiểm soát numerical quality. Reference rank≈31/32, giữ gần 100% variance ở 32 PCs; residual chỉ≈2,5–2,9×10⁻⁵, có thể nhạy tick/float32/near-null geometry. Numeric equivalence không chứng minh economic factor; feature distance không phát hiện concept drift. Cần kiểm tra parameter stability và incremental value ngoài các factors khác. Novelty chưa xác minh; oracle filtering, overlap, validation reuse và test đã xem vẫn tồn tại.

**Ghi chú: Các metrics proposed trong bảng benchmark là kịch bản minh họa, không phải kết quả thực nghiệm.**

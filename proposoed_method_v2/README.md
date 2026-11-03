# Paper gốc và bối cảnh của bốn proposed methods — v2

Ngày 03/11/2026. Bộ tài liệu đọc theo thứ tự: bản context này → một trong bốn methods. Chỉ chỉnh tài liệu; không chạy thí nghiệm mới, không thay MLP hoặc baseline. Không có biểu đồ.

| Tài liệu | Nội dung |
|---|---|
| [PDF paper gốc](applsci-15-11145.pdf) | Kuznetsov et al., Applied Sciences 2025, 15, 11145 |
| [Volatility-B](volatility_B.md) | Ước lượng reliability theo confidence và volatility |
| [Regime-B](market_regime_B.md) | Threshold theo trend và predicted direction |
| [ACF-A](autocorrelation_A.md) | Threshold theo độ mạnh autocorrelation |
| [Shift-A](distribution_shift_A.md) | Threshold theo khoảng cách khỏi training PCA subspace |

## 1. Paper gốc làm gì?

Paper: *Machine Learning Analytics for Blockchain-Based Financial Markets: A Confidence-Threshold Framework for Cryptocurrency Price Direction Prediction*, DOI [10.3390/app152011145](https://doi.org/10.3390/app152011145). Ý tưởng chính: **tách dự đoán hướng giá khỏi quyết định có thực hiện tín hiệu đó hay không**.

1. **Data:** macro daily OHLCV/technical indicators và micro minute-level order-book snapshots. Paper Table 2, tr.9 ghi macro 211.679 dòng/100 symbols; micro 5.672.947 dòng/11 symbols; unified 296 cột. Macro history 2018–2025; micro 10/2023–10/2024. Các counts của bảng là representations bài báo công bố, không phải raw schema hoặc số dòng BTC/ETH của chúng ta.
2. **Features:** daily MA, volatility, RSI; micro spread, imbalance, liquidity/depth và lag/rolling features. Đồng bộ macro đã trễ với micro; chọn 128 features bằng Mutual Information, tiền xử lý fit trên train (paper §§3.1–3.2; Table 3).
3. **Labels:** P là midpoint best bid/ask; future arithmetic return R=(P_future/P_t)−1. Up nếu R>δ, Down nếu R<−δ; phần ở giữa là Neutral/no-trade. Paper dùng δ=10 bps=0,1%, horizon 600 phút và huấn luyện binary trên directional samples (§§3.3–3.4, tr.13–15).
4. **Model:** MLP 128→256→128→64, ReLU, dropout 0,2, balanced BCE, Adam LR 0,001, batch 4096, budget 80 epochs, patience 10; chronological 70/15/15 riêng từng symbol. Paper mô tả isotonic calibration với 10-fold CV trong validation (Table 4; §4.2, tr.19).
5. **Selection:** p là calibrated probability Up, c=max(p,1−p). Model giữ hướng dự đoán; chỉ cho phép thực hiện khi c≥τ. τ được tối ưu **trên validation rồi đóng băng**, không phải số 0,8 đặt tùy ý. Grid paper 0,50–0,95 bước 0,01, tối đa mean net return/mẫu được chọn sau cost1 bps; paper báo nghiệm0,8 (Eq.6, tr.14; §3.4, tr.15).

$$
\tau^*=\arg\max_{\tau}\frac{\sum_{t\in V}\mathbf1(c_t\ge\tau)\,[u_tR_t^{bps}-1]}{\sum_{t\in V}\mathbf1(c_t\ge\tau)},\qquad \tau_t=\tau^*.
$$

Ở đây u=+1/−1 theo **predicted** Up/Down; V là validation. “Fixed” nghĩa là threshold không đổi theo thời điểm sau khi chọn, **vẫn có thể được tối ưu offline**. Chọn+Up→Long, chọn+Down→Short; không chọn→Abstain. Neutral là future label, không đồng nghĩa Abstain. Paper báo accuracy 82,68%, coverage 11,99%, mean net 151,11 bps trên executions (Table 11, tr.26); không gán các số đó cho reproduction BTC/ETH.

## 2. Phân biệt Paper / Repo / Baseline hiện tại

| Thành phần | Paper | Repo và baseline BTC/ETH đang so sánh |
|---|---|---|
| Universe | 11 pairs | BTC/ETH, unified1.008.203×296 |
| Labels | 10 bps, 600 phút | **100 bps=1%**, 600 **dòng/symbol**; dòng thiếu làm elapsed horizon thay đổi |
| MLP | PyTorch/Adam, dropout | Nhánh two-class dùng **sklearn/SGD**, hidden256–128–64, batch thực200, không dropout |
| Samples hiện tại | Counts riêng của paper | Train 292.338; validation 80.260; test 65.182, sau lọc Neutral |
| Calibration | Isotonic, CV trong validation | Prefit isotonic fit toàn validation |
| Threshold | τ*=0,8; grid bước 0,01 | τ*=0,64; grid 0,50–0,94 bước 0,02; pooled validation coverage≥10% |

Nguồn repo: [labels C:158](../centralized_training_two_classes.py#L158), [split C:602](../centralized_training_two_classes.py#L602), [lọc Neutral C:633](../centralized_training_two_classes.py#L633), [MLP C:1835](../centralized_training_two_classes.py#L1835), [calibration C:860](../centralized_training_two_classes.py#L860), [threshold optimization C:894](../centralized_training_two_classes.py#L894). Counts/config lấy từ baseline BTC/ETH và threshold đã khóa của run hiện tại. Báo cáo baseline đầu tiên ghi τ0,94 khi chưa có coverage constraint; không nhầm với baseline0,64 sau đó. Hướng dẫn dữ liệu, labels và lệnh chạy nằm riêng trong thư mục docs/.

## 3. Research gap: bốn methods muốn bổ sung điều gì?

Paper **đã dùng features thị trường và calibration**; không được nói tác giả bỏ qua volatility/trend hoàn toàn. Gap ở **selection layer**: một cutoff scalar chung không trực tiếp mô hình hóa reliability có điều kiện theo trạng thái thị trường hoặc khoảng cách tới training distribution. Có state features trong MLP không tự bảo đảm confidence đã calibrated như nhau ở mọi state.

| Giới hạn của selection rule cố định | Giả thuyết kiểm chứng | Method |
|---|---|---|
| Cùng c nhưng volatility khác có reliability khác | Volatility sửa mức nền và độ dốc confidence | Volatility-B |
| Cùng trend nhưng predicted Up/Down có payoff khác | Cho phép trend và direction điều chỉnh ngưỡng | Regime-B |
| Cùng c nhưng cấu trúc phụ thuộc returns khác | Một tilt theo absolute ACF giúp selection | ACF-A |
| Cùng c nhưng feature geometry xa train khác nhau | PCA residual bổ sung reliability information | Shift-A |

Đây là **hypotheses**, không phải bốn thiếu sót đã chứng minh của paper. Tác giả cũng nêu nhu cầu kiểm chứng qua market regimes và adaptive thresholds ở **§6.4, tr.28**. Chúng ta cụ thể hóa hướng đó bằng các đối chứng có thể bác bỏ giả thuyết; chưa xác minh novelty bằng literature review.

**Các methods không sửa được:** repo loại Neutral bằng future returns ở cả validation/test (oracle-filtered cohort); label overlap và thiếu purge main split; reuse validation; execution chỉ là signed sample-return rule, chưa quản lý vốn/vị thế; test đã xem. Đây là giới hạn của evaluation hiện tại, không được gán tất cả thành kết luận về mọi thí nghiệm trong paper. Không diễn giải các kết quả bên dưới thành deployable backtest hoặc confirmatory OOS.

## 4. Benchmark plan và mục tiêu so sánh

Giữ nguyên predictions/calibration/labels, cohort và phí 1 bps. Với e là selection mask, k=1 nếu predicted direction đúng, N là eligible samples và n=Σe:

$$
g_t=u_tR_t^{bps}-1,\quad C=\frac nN,\quad A_{sel}=\frac{\sum e_tk_t}{n},\quad
\bar g=\frac{\sum e_tg_t}{n},\quad U=\frac{\sum e_tg_t}{N}=C\bar g.
$$

Cách đọc: coverage, selective accuracy, net/mẫu chọn và expected net/mẫu đủ điều kiện. n=0: accuracy và net/chọn undefined, U=0 nếu N>0. Counts là signals, không phải vị thế độc lập. Không cộng overlapping returns thành portfolio PnL hoặc annualized Sharpe.

- **Selection:** coefficients/κ chỉ fit full validation. ACF/Regime/Shift tối đa mean net/chọn trừ regularization; Volatility-B fit correctness loss rồi chọn κ theo mean net/chọn. Coverage≥10% chỉ áp validation pooled; không bắt test hoặc từng coin đạt 10%.
- **Coverage control:** ngoài native results, chọn cutoffs bằng validation counts cho targets10%,20%,25,50%,40%, đóng băng sang test; không dùng test quantiles/top-k. Confidence plateaus có thể làm target infeasible. Báo actual coverage/drift; chỉ gọi near-matched nếu actual coverages thực sự gần nhau, không lấy target thay actual.
- **Attribution:** BTC/ETH, predicted Up/Down, asset×direction và tuần; matching trong asset×direction; ablation factor-free. Temporal diagnostics fit/chọn trên past prefixes, purge theo label endpoint thật, báo infeasible. Validation đã dùng bởi model/calibrator nên đây không phải independent holdout.

Nguồn paper: [MDPI](https://mdpi-res.com/d_attachment/applsci/applsci-15-11145/article_deploy/applsci-15-11145.pdf), DOI [10.3390/app152011145](https://doi.org/10.3390/app152011145). Kuznetsov et al., Applied Sciences 2025. PDF công bố theo CC BY4.0.

"""Render empirical report from frozen training references and validation-only outputs."""
import json
import numpy as np
import pandas as pd
from build_scores import ROOT,OUT

def table(x):
    return '| '+' | '.join(map(str,x.columns))+' |\n|'+'|'.join(['---']*len(x.columns))+'|\n'+'\n'.join('| '+' | '.join('undefined' if pd.isna(v) else str(v) for v in row)+' |' for row in x.itertuples(index=False,name=None))

def percent(x,cols):
    x=x.copy()
    for c in cols:x[c]=(100*x[c]).round(2)
    return x

def main():
    audit=pd.read_csv(OUT/'reference_audit.csv')
    distribution=pd.read_csv(OUT/'score_distributions.csv')
    q=pd.read_csv(OUT/'quantile_metrics.csv')
    summary=pd.read_csv(OUT/'incremental_summary.csv')
    effects=pd.read_csv(OUT/'conditional_strata_effects.csv')
    health=pd.read_csv(OUT/'control_model_health.csv')
    status=json.loads((OUT/'score_status.json').read_text());conditional=json.loads((OUT/'conditional_status.json').read_text())
    primary=['md128_l01','knn20_pc16','psi360_pc3','w1_360_pc1']
    d=distribution[distribution.score.isin(primary+['pca_re32'])][['score','symbol','missing','median','p99','max','train_q99','share_above_train99']]
    d=percent(d,['share_above_train99']).round(5)
    quant=q[q.score.isin(primary)&q.scope.isin(['BTC_USDT','ETH_USDT'])][['score','scope','quantile','n','confidence_mean','accuracy','selected_n','coverage','selective_accuracy','net_selected_bps','ece_10']]
    quant=percent(quant,['accuracy','coverage','selective_accuracy','ece_10']).round(3)
    pooled=q[(q.scope=='pooled_within_symbol_quantiles')&q.score.isin(primary)&q['quantile'].isin([1,5])][['score','quantile','n','selected_n','coverage','selective_accuracy','net_selected_bps','expected_net_bps']]
    pooled=percent(pooled,['coverage','selective_accuracy']).round(3)
    increments=summary[summary.scope=='pooled'][['score','positive_logloss_folds','gain_logloss','gain_brier','positive_rmse_folds','gain_rmse_net','effect_correct_pp','effect_net_bps']].round(5)
    partial=effects[(effects.spec=='strata_day')&effects.score.isin(primary+['pca_re32'])][['score','symbol','partial_corr_correct','partial_corr_net','effect_correct_pp_per_IQR','effect_net_bps_per_IQR']].round(3)
    extreme=pd.read_csv(OUT/'training99_cohorts.csv')
    extreme=extreme[extreme.score.isin(['md128_l01','knn20_pc16'])&extreme.scope.isin(['BTC_USDT','ETH_USDT'])][['score','scope','above_train99','n','accuracy','selected_n','selective_accuracy','net_selected_bps']]
    extreme=percent(extreme,['accuracy','selective_accuracy']).round(3)
    temporal=pd.read_csv(OUT/'temporal_incremental_metrics.csv')
    stability=temporal[(temporal.scope=='pooled')&temporal.score.isin(['pca_re32','knn20_pc16','md128_l01','w1_360_pc1'])][['fold','score','gain_logloss','gain_brier','gain_rmse_net']].round(6)
    text=r'''# Distribution Shift / OOD Exploratory Analysis — BTC/ETH validation

Ngày 03/11/2026. **CONDITIONAL GO cho kiểm chứng tiếp về geometry/reference; chưa đủ cơ sở xây OOD-adaptive threshold.** Không có quy luật chung “xa train hơn thì model sai hơn”. PCA reconstruction residual có incremental gain nhỏ tương đối nhất quán, nhưng có giới hạn rank/precision/reference cần kiểm tra và chưa chứng minh economic utility. Mahalanobis/kNN/rolling PSI/Wasserstein chưa tạo evidence ổn định để thay Fixed.

Chỉ phân tích validation; training features chỉ dùng xây reference. Không đọc test predictions/arrays, không dùng training labels để fit reference, không train lại MLP, không đề xuất/triển khai adaptive rule. Fixed τ=0,64, labels/probabilities/directions/cost=1 bps và artifacts cũ giữ nguyên.

## 1. Dữ liệu và integrity

Sử dụng đúng **292.338×128 X_train** và **80.260×128 X_validation** float32 đã lưu, theo feature order của MLP. Đây là output median imputer + RobustScaler (quantile_range 25–75) fit trong baseline; không fit lại baseline preprocessors. Train metadata chỉ đọc symbol/timestamp, không đọc y_train. Không lấy validation/test làm reference, không subsample training reference cho kNN. SHA256 X_train/X_validation khớp matrix_manifest.

Full validation vẫn BTC 40.209, ETH 40.051. Fixed chọn 20.463 mẫu, coverage 25,50%, selective accuracy 68,93%, net 99,18 bps/mẫu chọn. Validation keys đối chiếu metadata/matrix order; train/validation keys không trùng. Model probabilities và labels lấy từ artifacts đã lưu.

Nguồn: [original transform](centralized_training_two_classes.py#L545), [preprocessor fit](centralized_training_two_classes.py#L496), [training wrapper](reproduction_setup/train_btc_eth_1pct.py), [matrix manifest](data/btc_eth_1pct/prepared/matrix_manifest.json), [feature schema](data/btc_eth_1pct/prepared/feature_schema.json).

### Reference riêng từng asset và thời gian

'''+table(audit[['symbol','train_n','val_n','train_start','train_end','val_start']])+r'''

Mỗi primary reference dùng toàn bộ training features của đúng coin, kết thúc trước validation của coin đó. Pooled train kết thúc **25/06/2024 19:40 UTC**, muộn hơn validation BTC bắt đầu: có **1.905 early BTC validation rows** trước/ở global train end. `pooled_md128` chỉ là sensitivity có temporal caveat, **không dùng trong predictive incremental models**. Per-symbol reference tránh thêm lookahead từ pooling reference; không khắc phục các vấn đề global preprocessing/MI/model split đã kế thừa từ repo.

Reference training cohort cũng đã được lọc Neutral theo future returns ở pipeline gốc. Vì vậy đây là P(X|cohort gốc), không phải reference mọi trạng thái thị trường. Rolling windows bên dưới dùng mọi unified observation quá khứ; sự khác nhau giữa population reference và rolling population là một giới hạn cần nhớ.

## 2. Geometry, scores và công thức

### 2.1. Scaling chỉ cho phân tích

Gọi x là vector 128 chiều **đã RobustScaled như MLP**. Với symbol s, fit mean/std từ X_train,s:

$$
z_s(x)=D_s^{-1}(x-\mu_s),\qquad
S_s=\frac1{N_s-1}\sum_i z_s(x_i)z_s(x_i)^\top.
$$

Cách đọc: thêm train-only standardization riêng coin để geometry không bị một feature có variance lớn áp đảo. Cột train std≤10⁻¹² đặt scale=1, vẫn giữ cột; thực tế không có cột constant ở reference này. **Đây là analysis geometry, không đổi input hoặc scaler của MLP.** Tính toán reference float64 trên đúng float32 cached values; không giảm precision dữ liệu. Sensitivity `rawspace_md128` bỏ bước standardization thứ hai, dùng nguyên MLP-scaled space; tên “rawspace” không có nghĩa là unscaled dollar-price space.

### 2.2. Mahalanobis có shrinkage, 128 chiều

$$
S_{s,\lambda}=(1-\lambda)S_s+\lambda\bar v_s I,\quad
\bar v_s=\operatorname{tr}(S_s)/128,\quad
MD_{s,\lambda}(x)=\sqrt{z_s(x)^\top S_{s,\lambda}^{-1}z_s(x)}.
$$

Cách đọc: khoảng cách có điều chỉnh covariance; λ=0,01 primary và 0,10 sensitivity, không tune theo reliability. Tính qua eigenbasis, không invert plain covariance gần singular. Clamp eigenvalues âm rất nhỏ do numerical roundoff về 0 trước cộng shrinkage; không sửa giá trị X. [Scikit-learn covariance documentation](https://scikit-learn.org/stable/modules/covariance) mô tả shrinkage về scaled identity. Không gọi đây là Ledoit–Wolf tự ước lượng vì λ ở đây cố định.

Không giả định Gaussian đủ đúng để chuyển MD thành xác suất OOD hoặc p-value chi-square. Fit covariance O(Nd²), scoring O(Md²) qua full eigenbasis; d=128 vừa với máy hiện tại, reference fit theo coin để kiểm soát RAM. RobustScaler không loại mọi influence của extreme reference points lên sample covariance.

### 2.3. PCA distances và residual

Fit eigenvectors U và eigenvalues e từ **training covariance**. Với a=Uᵀz, r∈{16,32}:

$$
D_{PCA,r}(x)=\sqrt{\sum_{j=1}^r\frac{a_j^2}{0.99e_j+0.01\bar v}},\qquad
RE_{32}(x)=\sqrt{\frac1{128-32}\sum_{j=33}^{128}a_j^2}.
$$

Cách đọc: novelty trong retained subspace và residual RMS ngoài subspace là hai khái niệm khác nhau. PCA có thể bỏ hướng variance nhỏ nhưng quan trọng cho reliability; RE là complementary score, không phải confidence uncertainty được calibrated. Fit như covariance/eigendecomposition; projection r chiều O(Mdr).

### 2.4. kNN distance

$$
KNN_k(x)=\frac1k\sum_{j=1}^k\|a_{1:16}(x)-a_{1:16}(x^{train}_{NN_j})\|_2,
\qquad k\in\{5,20\}.
$$

Cách đọc: mean distance đến k neighbours của cùng coin, **toàn bộ** 131.442/160.896 training points, PCA16 không whiten; query exact cKDTree, eps=0, hai workers. Không dựa vào nhãn train. Cùng query k=20 tính cả hai scores; query theo batch. Worst-case high-dimensional search có thể gần brute-force; PCA giảm chiều và training shape thấp-rank giúp bài toán này khả thi. Không gọi score này là LOF: không có local-density ratio như [novelty/outlier documentation](https://scikit-learn.org/stable/modules/outlier_detection).

Để mô tả train-distance Q99, dùng **2.048 evenly-spaced training anchors/coin** với leave-one-out neighbour distances. Chỉ calibration summary này dùng anchors; reference tree không bị subsample. Đây không phải independent deployment false-positive calibration vì neighbours training gần nhau theo thời gian.

### 2.5. Rolling PSI và exact Wasserstein

Dùng 360 observations liên tiếp của full past unified stream, gồm cả các rows không nằm trong oracle-filtered prediction cohort. Raw selected features được transform bằng **imputer/scaler cũ**, cast float32 đúng baseline, rồi chiếu qua training reference. **Toàn bộ 292.338 train và 80.260 validation transformed rows đã được đối chiếu exact equality với cached matrices: PASS.** Không chỉ so số hàng. PCA projections, histograms và W1 không fit lại reference từ validation.

PSI trên từng PC1–PC3, bins là training deciles; outer edges ±∞, duplicate edges bỏ; pseudocount 0,5:

$$
p_{jb}=\frac{n^{train}_{jb}+0.5}{N_s+0.5B_j},\quad
q_{tjb}=\frac{n^{past360}_{tjb}+0.5}{360+0.5B_j},\quad
PSI_t=\frac13\sum_{j=1}^{3}\sum_{b=1}^{B_j}(q_{tjb}-p_{jb})\log(q_{tjb}/p_{jb}).
$$

Cách đọc: khác biệt marginal histogram hiện tại và train, trung bình ba PCs; không đo toàn joint distribution. PC/bins/p đều chỉ fit trên training. Chi phí O(N_stream×3B), không fit model hoặc threshold. Không áp dụng các mốc PSI 0,1/0,25 như significance hoặc OOD ground truth.

$$
W_t=\frac{\int_0^1|Q^{train}_{PC1}(u)-Q^{past360}_{PC1,t}(u)|du}
{IQR^{train}_{PC1}}.
$$

Cách đọc: exact empirical one-dimensional Wasserstein-1 giữa toàn training PC1 và rolling PC1, chia training IQR. Không phải multidimensional Wasserstein hoặc quantile-grid approximation. Dùng integrated empirical training quantile và sorted 360-point windows, query batches 512; đối chiếu scipy trên actual windows PASS. Công thức empirical W1 tham khảo [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html). Chuẩn bị training O(N log N), mỗi window khoảng O(W log W + W log N), W=360; tránh sort lại toàn reference cho mỗi query.

Train Q99 của W1 dùng 2.048 past training-anchor windows/coin; train PSI quantiles dùng mọi eligible training-row window. Đó là diagnostic references, không phải cutoffs trading. Cửa sổ gồm t sau khi aggregate t đã có, không future information. Q99 train được dùng để phân tích novelty; **không chọn threshold execution**.

### 2.6. Marginal tail fraction

Score bổ sung đơn giản: fraction của 128 features ngoài train 1st–99th percentile range, fit riêng coin. Bounded [0,1], O(Md). Không mô hình covariance/joint support; correlated features có thể đếm trùng cùng một price-level move. Không xóa hoặc winsorize outliers.

## 3. Covariance singularity, dimensions và score health

'''+table(audit[['symbol','constant_columns','numerical_rank','max_eigen','retained16','retained32','regularized_condition01']].round(8))+r'''

Numerical rank ở tolerance eigenvalue>10⁻¹⁰×largest eigenvalue chỉ 31 BTC/32 ETH, dù 128 cột đều có variance. PCA16 giữ 99,971%/99,923% variance; PCA32 gần 100%. Không khẳng định algebraic rank chỉ 31/32 ở mọi precision, nhưng **plain inverse covariance rất không ổn định**. First PC eigenvalue khoảng 115 trên trace khoảng 128, tức gần 90% reference variance. Top loadings đều là các orderbook price levels gần trùng nhau: xem [PCA_loadings.csv](research_distribution_shift/validation_20261103/PCA_loadings.csv). Đây có thể chủ yếu là price-level geometry, không đại diện mọi risk factors thị trường.

RE32 validation median khoảng 2,5×10⁻⁵ BTC/2,9×10⁻⁵ ETH trong standardized units. Một score nhỏ không tự là noise; nhưng cần phân biệt residual cấu trúc orderbook, tick/float32 quantization và sensitivity của low-variance subspace. Chưa xác minh nó là lỗi numerical hoặc một signal tài chính thật.

## 4. Phân phối và quantile performance

### 4.1. Train-reference distances khác mức error

Cột share_above_train99 là %, chỉ là diagnostic exceedance, không phải ground-truth OOD rate. Missing bằng 0 ở cả 12 scores. Train KNN/W1 Q99 dùng anchors như đã nêu. [score_distributions.csv](research_distribution_shift/validation_20261103/score_distributions.csv) giữ min/p01/p90/p95/p99/max, outlier flags và tất cả scores.

'''+table(d)+r'''

kNN20 vượt train Q99 ở **99,30% BTC/98,27% ETH**. Train LOO neighbours có thể là observations gần/giống nhau trong thời gian, còn validation không có những neighbours thời gian đó trong frozen reference; đây là calibration/temporal-neighbour mismatch hợp lý cần kiểm tra, không kết luận 99% samples sai hoặc truly OOD.

PSI khoảng 5,84 ở cả train và validation; khoảng 35,0% BTC/44,9% ETH validation có PSI>5,8. Window 6 giờ thường gần một train quantile bin trên mỗi PC, còn train reference trải nhiều tháng; PSI có thể lớn ngay cả khi giá nằm sâu trong historical support. Khi q tập trung một bin và p gần uniform, score bão hòa và ít phân biệt trạng thái. W1 validation median chỉ khoảng 0,51, dưới train-window Q99 khoảng 1,1: các scores không đo cùng một khái niệm novelty.

### 4.2. Quantiles validation theo coin

Quantiles dùng validation chỉ để mô tả, không xây reference. Ties giữ cùng nhóm, duplicate edges được bỏ; **không đảm bảo nhóm đúng 20%**, đặc biệt PSI/tail fraction. Các cột accuracy/coverage/selective accuracy/ECE là %, confidence_mean là probability. Accuracy/ECE trên toàn nhóm; selective metrics chỉ trên c≥0,64; net là bps sau 1 bps phí. Zero selections → undefined, không ghi accuracy=0 hoặc tự tăng coverage.

'''+table(quant)+r'''

BTC MD Q3 chỉ **6 selections**, ETH MD Q1 **0 selections**, ETH W1 Q5 **11 selections**: không kết luận lợi thế/hại từ những ô này. Hiệu ứng thay đổi coverage rất mạnh, nên không đọc raw selective accuracy như so sánh cùng lượng cơ hội.

### 4.3. Tổng hợp theo within-symbol quantiles

Pooled groups gộp quantile riêng mỗi coin, không định nghĩa global score cutoff trộn hai reference. Nếu ties tạo nhóm lệch size/số nhóm, asset proportions vẫn có thể khác; đây không phải coverage-matched benchmark mới.

'''+table(pooled)+r'''

![Quantile performance](research_distribution_shift/validation_20261103/quantile_performance.png)

### 4.4. Confidence và calibration

Confidence min/median/P90/P95/max và ECE10 đều lưu trong quantile_metrics.csv; ECE là weighted |correctness−confidence| ở 10 uniform bins [0,1], top-label calibration, không phải probability Up calibration. ETH MD Q5 mean confidence 0,6492 nhưng accuracy toàn nhóm 57,92%, selective accuracy 58,05%; BTC MD Q5 mean confidence 0,6310, accuracy 56,91%, selective 70,22%. Confidence distributions khác nhau giữa shift groups, nên cần controls, không chỉ nhìn error rates.

Calibrator gốc đã fit trên validation; các reliability diagrams này không độc lập. Bins trong hình chỉ hiển thị khi n≥100; dữ liệu mọi bins được giữ lại.

![Reliability by shift](research_distribution_shift/validation_20261103/reliability_by_shift.png)

## 5. Counterexamples: shift lớn vẫn đúng, shift thấp vẫn sai

'''+table(extreme)+r'''

MD above train Q99 ở BTC có accuracy toàn nhóm 58,04%, selective 76,05%, net +416,85 bps/mẫu chọn; ETH cùng loại exceedance có accuracy 44,36%, selective 44,29%, net −128,28. Không có universal rule “OOD thì abstain”. Trong low kNN-distance (≤train Q99), BTC 283 rows đều đúng nhưng ETH 691 rows chỉ 0,87% đúng; **cả hai nhóm không có selection tại τ=0,64**. Low novelty không bảo đảm correctness, đồng thời baseline confidence đã abstain ở một số nhóm đó. Các groups này vẫn oracle-filtered/serially dependent.

RE32 above train Q99 có raw accuracy cao hơn phần còn lại ở cả hai coin (BTC 78,28%, ETH 62,41%); ETH chỉ 186 selections ở nhóm này, phải gắn cờ ít mẫu. Đây còn phản bác cách đọc đơn giản “reconstruction error lớn chắc model sai”, dù adjusted/temporal model effects bên dưới có thể khác.

## 6. Incremental information sau các factors đã nghiên cứu

### 6.1. Statistical comparators và temporal protocol

Training baseline features **chỉ xây reference**, không dùng training labels để fit các reliability comparators. Các models thống kê sau fit từ **past validation prefixes**, đánh giá subsequent validation intervals; không thay hoặc train lại MLP.

M0 correctness: logistic L2 C=0,01; return: ridge alpha=10. Controls: confidence logit và exact symbol×confidence×direction strata (rare<20 prefix rows gộp OTHER); symbol/direction; confidence×symbol/direction; log micro/daily volatility; signed/absolute tanh(trend), Bull/Bear indicators; signed/absolute ACF1_360, signed ACF5_1440 và ER360. Linear calendar predictor của phân tích trước được **bỏ trước khi xem shift outcomes**, để tránh extrapolation đã không ổn định; không tune C/alpha sau kết quả.

M1 thêm log1p(score), standardize chỉ trên train prefix. 11 scores primary/sensitivity causal được kiểm tra riêng; pooled_md128 bị loại vì temporal caveat. Thêm bốn score×ETH diagnostics cho MD/kNN/PSI/W1, không search nhiều combinations để tạo edge. Tất cả normalization/encoders/statistical coefficients fit prefix-only. Score reference vẫn **fixed training-only**, không xây lại bằng validation prefix.

Ba expanding folds giữ ranh giới chung BTC/ETH và purge theo future endpoint thật của label 600 dòng; eval sizes 19.927,19.061,23.708, tổng 62.696. Phần full-cohort descriptive vẫn đủ 80.260 mẫu, không đổi baseline scope. MLP/calibrator gốc đã dùng validation nên đây không phải independent OOS validation của MLP.

### 6.2. Comparator health

'''+table(health.round(5))+r'''

M0 vẫn kém confidence reference ở cả 3 folds; fold đầu mean correctness dự báo 72,25% so với actual 38,52%. Có thể còn temporal relationship shift/overfit controls; không khẳng định nguyên nhân đã được nhận diện. Reference confidence đã calibrated trên toàn validation nên cũng có advantage reuse; không gọi đó là independent calibration baseline. **Gain so với M0 yếu chỉ là relative evidence**, không chứng minh một deployable reliability predictor tốt.

### 6.3. Predictive increments và effect sizes

Gain = loss M0−loss M1, dương tốt hơn. Summary weighted theo evaluation counts; RMSE là weighted mean của fold-RMSE reductions, không global pooled RMSE. Model-implied effect là trung bình thay đổi predicted correctness (điểm phần trăm) hoặc net (bps) khi log1p(score) từ prefix Q25 sang Q75, giữ controls khác cố định. Không phải causal effect hoặc lợi nhuận strategy; với asset interaction, score×ETH được thay đồng thời.

'''+table(increments)+r'''

**RE32**: Δlogloss +0,00453, ΔBrier +0,00152, mean fold-return-RMSE reduction +0,110 bps; logloss cải thiện cả 3 folds ở cả BTC và ETH. Đây là candidate có relative prediction evidence nhất quán nhất trong tập đã thử, **nhưng hiệu ứng return rất nhỏ**, comparator yếu và geometry rank/precision cần kiểm tra. Không coi nó là OOD score đã validated để giao dịch.

**kNN20**: Δlogloss pooled +0,00252 nhưng BTC −0,02290 và ETH +0,02675; chỉ 2/3 pooled folds dương, return-RMSE average giảm chất lượng khoảng 0,101 bps. Không có improvement chung hai coin. **MD128 λ=0,01**: Δlogloss −0,09774, cả 3 pooled folds âm; PCA16/32 cũng không cải thiện. PSI/W1 average âm, interactions không cứu được generalisation.

Fold breakdown một số scores:

'''+table(stability)+r'''

![Incremental effects](research_distribution_shift/validation_20261103/incremental_effects.png)

### 6.4. Conditional strata và day robustness

Kiểm tra riêng asset bằng fixed-effects demean trong confidence×direction×vol-quintile×trend-quintile×|ACF|-quintile strata; thêm day cho spec dưới. Partial out continuous log vol/trend/ACF còn dư, bỏ daily control khi đã bị day effects hấp thụ. Score dùng log1p. Effect theo **full-validation IQR**, không cùng estimator với prefix counterfactual ở trên. Global IQR có thể lớn hơn within-stratum variation; coefficients tuyến tính có thể phóng đại hypothetical change, nên báo partial correlation và không diễn giải probability forecast/causal effect.

'''+table(partial)+r'''

W1 vẫn có positive conditional correctness/net association sau controls/day, đặc biệt ETH, trong khi predictive increments theo thời gian không ổn định và average âm. Đây là khác biệt giữa **association trong cùng validation** và **incremental predictive value qua periods**. kNN effect mạnh trong no-day strata (BTC −9,33 điểm phần trăm/IQR; ETH +25,56) yếu đi/đổi khi thêm ngày. RE32 sau day có correctness effects chỉ khoảng −0,22/−0,20 điểm phần trăm mỗi IQR. Không có bằng chứng score monotonic error risk dùng chung BTC/ETH.

## 7. Sensitivity, volatility/trend correlation và thời gian

'''+table(pd.read_csv(OUT/'sensitivity_correlations.csv').round(5))+r'''

Regularization λ=0,01/0,10 và k=5/20 cho rank correlations cao, nhưng **score-rank robustness không đồng nghĩa predictive-utility robustness**. Rawspace-vs-extra-standardized MD khác hơn ở ETH (rho≈0,932) so với BTC (0,996). PCA16/32 rho≈0,982/0,966; MD128 và PCA32 cho temporal results gần như giống nhau, phù hợp phần variance bị bỏ rất nhỏ. Không chọn một dimension/λ có một ô accuracy đẹp để tune baseline.

RE32 tương quan Spearman với micro volatility khoảng +0,421 BTC/+0,338 ETH; nó không độc lập volatility chỉ vì fit từ features. Score–volatility/trend/ACF/confidence/mid-price correlations đầy đủ trong control_correlations.csv. PCA first components tập trung price levels; PSI/W1 có thể phản ánh local price concentration hoặc vị trí trong long historical range thay vì model uncertainty.

weekly_quantiles.csv giữ mọi tuần có mẫu và flags cho selections<200; không bỏ episode xấu. Các forward metrics cho thấy MD kém 3/3 folds, kNN/rolling scores đổi dấu lợi ích; không kết luận stability từ pooled sample count. Các window overlaps 360 dòng, labels 600 dòng, neighbours/rows training phụ thuộc thời gian và quantile groups có thể tập trung một episode. Chưa làm IID significance hoặc coi 80.260 rows là số thử nghiệm độc lập; nếu cần CIs phải có paired time-block design và đủ episodes.

![Temporal scores](research_distribution_shift/validation_20261103/temporal_shift.png)

## 8. Novelty, covariate shift và concept drift

| Khái niệm | Đã đo được gì | Không được kết luận |
|---|---|---|
| Pointwise novelty | MD/PCA/kNN/tail distance của một observation so với train geometry | Khoảng cách là xác suất prediction error hoặc ground-truth OOD |
| Feature-distribution shift | PSI/W1 của past window so với train distribution trong projected marginal spaces | Đã phát hiện mọi joint shift hoặc đã có một statistically calibrated detector |
| Covariate shift theo định nghĩa chặt | Cần P(X) thay đổi nhưng P(Y given X) không đổi | Chỉ feature distances chưa xác minh conditional invariance |
| Concept drift | Cần evidence quan hệ P(Y given X) thay đổi | Feature distance hoặc error rate thay đổi đơn lẻ chứng minh concept drift |

Định nghĩa covariate shift với conditional output distribution giữ nguyên theo [Sugiyama, Krauledat và Müller, JMLR 2007](https://jmlr.org/papers/volume8/sugiyama07a/sugiyama07a.pdf). Báo cáo này chỉ có evidence feature-space novelty/distribution differences và reliability associations; **không tuyên bố phát hiện concept drift**, cũng không chứng minh các thay đổi thuộc covariate-shift-only setting. Performance có thể đổi do labels/class balance, oracle cohort, calibration reuse, tail events hoặc conditional relation; chưa phân rã đủ.

## 9. Trả lời câu hỏi và quyết định nghiên cứu

1. **Score liên quan mạnh nhất tới errors?** Không có monotonic winner. RE32 có conditional/temporal relative evidence nhỏ nhưng nhất quán hơn; MD/kNN raw error relationships khác asset. Khoảng cách lớn không mặc định error cao.
2. **Thông tin ngoài confidence và factors trước?** RE32 có small relative gains sau controls ở cả hai coin/ba folds; chưa đủ chứng minh useful deployable prediction. kNN có pooled gain nhỏ nhưng heterogeneity mạnh. W1 conditional effect còn nhưng không tổng quát hóa ổn định qua periods.
3. **Ổn định BTC/ETH và theo thời gian?** Chưa đủ cho phần lớn scores. Score rankings khá robust với k/λ, nhưng performance increments khác nhau. Reliability controls cũng còn failure theo thời gian, hạn chế mức tin cậy của kết luận.
4. **Đáng phát triển adaptive threshold?** **CONDITIONAL GO cho kiểm chứng tiếp**, chưa GO triển khai threshold. Trước hết cần kiểm chứng ý nghĩa RE32/precision/reference và density calibration theo thời gian. Không thêm score phức tạp chỉ vì MD/PSI không cho kết quả mong muốn.
5. **Ba hypotheses:** H1: reconstruction residual chứa reliability information về orderbook shape/microstructure ngoài price-level factor, nhưng có thể nhạy tick/float32 quantization; cần phân biệt hai khả năng. H2: kNN “OOD” prevalence cực cao chủ yếu do temporal-neighbour/reference calibration mismatch, không phải tương ứng error risk. H3: rolling distribution scores chủ yếu đo short-window concentration so với long historical reference và period composition; liên hệ utility không chắc còn khi đổi period/asset/controls. Chưa thiết kế công thức adaptive để kiểm chứng các hypotheses này.

Không chọn method/threshold hoặc đọc test để tìm winner ở bước này. Training reference vẫn cùng oracle-filtered baseline BTC/ETH; không suy rộng sang toàn 11 crypto hoặc exact paper reproduction.

## 10. Artifacts và tái lập

- [Reference audit](research_distribution_shift/validation_20261103/reference_audit.csv), [protocol](research_distribution_shift/validation_20261103/protocol.json), [score status](research_distribution_shift/validation_20261103/score_status.json): full train/validation transform equality và direct PSI/exact W1 checks PASS. [Integrity checks](research_distribution_shift/validation_20261103/integrity_checks.json) xác minh lại cuối lượt toàn bộ hashes source/model/reference inputs không đổi; không đọc test predictions để kiểm tra.
- [Distributions](research_distribution_shift/validation_20261103/score_distributions.csv), [all quantile metrics](research_distribution_shift/validation_20261103/quantile_metrics.csv), [direction](research_distribution_shift/validation_20261103/direction_quantiles.csv), [weekly](research_distribution_shift/validation_20261103/weekly_quantiles.csv), [train-Q99 cohorts](research_distribution_shift/validation_20261103/training99_cohorts.csv).
- [Incremental summary](research_distribution_shift/validation_20261103/incremental_summary.csv), [per-fold metrics](research_distribution_shift/validation_20261103/temporal_incremental_metrics.csv), [conditional-strata effects](research_distribution_shift/validation_20261103/conditional_strata_effects.csv), [control health](research_distribution_shift/validation_20261103/control_model_health.csv), [sensitivity](research_distribution_shift/validation_20261103/sensitivity_correlations.csv).
- training_references.joblib giữ geometry/PCA/full kNN coordinates từ train, không labels; validation_scores.parquet/validation_analysis_rows.parquet giữ scores ghép validation; past_PC_stream.parquet chỉ là projected feature stream, không test predictions. statistical_controls/ và statistical_validation_predictions.parquet thuộc các regression chẩn đoán validation, không thay MLP artifacts.
- Chạy lần lượt `reproduction_setup/venv/Scripts/python.exe research_distribution_shift/build_scores.py`, rồi `describe.py`, `conditional_analysis.py`, `robustness.py`, `write_report.py` với cùng thư mục research_distribution_shift/. Script build và statistical model folder từ chối overwrite. Để tái lập vào thư mục khác, đổi OUT trong build_scores trước; không xóa outputs cũ. Không cài thư viện hoặc tải dữ liệu.

'''+f"Reference/score build {status['elapsed_sec']:.2f}s, statistical comparators {conditional['elapsed_sec']:.2f}s, peak working set {status['memory']['peak_ws_gib']:.3f}/{conditional['memory']['peak_ws_gib']:.3f} GiB; chưa gồm describe/robustness/report. Guard stage peak WS≤9 GiB, available≥2 GiB.\n"
    import re
    text=re.sub(r'(!?\[[^\]]*\]\()([^)]+)(\))', lambda m: m.group(1)+(m.group(2) if m.group(2).startswith(('http://','https://','mailto:','#')) else '../'+m.group(2))+m.group(3), text)
    (ROOT/'analysis/DISTRIBUTION_SHIFT_EXPLORATORY_ANALYSIS.md').write_text(text,encoding='utf-8')
    print('Wrote DISTRIBUTION_SHIFT_EXPLORATORY_ANALYSIS.md')

if __name__=='__main__':main()

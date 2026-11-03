"""Write report exclusively from saved validation analyses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from explore import ROOT,OUT

def table(df):
    return '| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join('undefined' if pd.isna(v) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def pct(df,cols):
    df=df.copy()
    for c in cols:df[c]=(100*df[c]).round(2)
    return df

def main():
    distribution=pd.read_csv(OUT/'distributions.csv')
    quantiles=pd.read_csv(OUT/'quantile_metrics.csv')
    summary=pd.read_csv(OUT/'incremental_summary.csv')
    temporal=pd.read_csv(OUT/'temporal_incremental_metrics.csv')
    strata=pd.read_csv(OUT/'conditional_strata_effects.csv')
    folders=pd.read_csv(OUT/'temporal_folds.csv')
    health=pd.read_csv(OUT/'control_model_health.csv')
    primary=['acf_1_360','abs_acf_1_360','vr_5_360','er_360']
    dist=distribution[distribution.feature.isin(primary)][['feature','symbol','missing','minimum','q25','median','q75','maximum','tukey_outlier_share']].copy()
    dist=dist.round(5)
    q=quantiles[quantiles.feature.isin(primary)][['feature','symbol','quantile','n','confidence_mean','accuracy','selected_n','coverage','selective_accuracy','net_selected_bps','expected_net_bps','ece_10']].copy()
    q=pct(q,['accuracy','coverage','selective_accuracy','ece_10']).round(3)
    signed=pd.read_csv(OUT/'acf_sign_bands.csv')
    signed=signed[signed.feature=='acf_1_360'][['symbol','sign_band','n','share','confidence_mean','accuracy','selected_n','coverage','selective_accuracy','net_selected_bps','ece_10']]
    signed=pct(signed,['share','accuracy','coverage','selective_accuracy','ece_10']).round(3)
    main_summary=summary[summary.scope=='pooled'][['feature','positive_logloss_folds','gain_logloss','gain_brier','positive_rmse_folds','gain_rmse_net','gain_mae_net','effect_correct_pp','effect_net_bps']].round(5)
    select=['acf_1_360','abs_acf_1_360','acf_5_1440','abs_acf_1_1440','er_60','vr_5_360']
    cond=strata[strata.feature.isin(select)][['feature','symbol','spec','partial_corr_correct','effect_correct_pp_per_IQR','effect_net_bps_per_IQR']].round(3)
    effect=temporal[(temporal.scope=='pooled')&temporal.feature.isin(['acf_5_1440','abs_acf_1_1440','er_60'])][['fold','feature','gain_logloss','gain_brier','gain_rmse_net','effect_correct_pp','effect_net_bps']].round(5)
    direction=pd.read_csv(OUT/'direction_quantiles.csv')
    dr=direction[(direction.feature=='acf_1_360')&direction['quantile'].isin([1,5])][['symbol','quantile','direction','n','selected_n','selective_accuracy','net_selected_bps']]
    dr=pct(dr,['selective_accuracy']).round(3)
    interactions=temporal[(temporal.feature=='acf_1_360_interactions')&temporal.scope.isin(['predicted_Up','predicted_Down'])][['fold','scope','effect_correct_pp','gain_logloss','gain_rmse_net']].round(5)
    statuses=[json.loads((OUT/f).read_text()) for f in ['exploration_status.json','conditional_status.json']]
    text=r'''# Autocorrelation / Market Efficiency Exploratory Analysis

Ngày 03/11/2026. **Kết luận: CONDITIONAL GO cho nghiên cứu tiếp; chưa đủ evidence để xây hoặc thay thế baseline bằng adaptive threshold.** Có residual associations nhưng dấu/độ lớn phụ thuộc asset, thời gian và mô hình kiểm soát. Những chênh lệch raw rất lớn không chuyển thành incremental prediction value ổn định. Chưa đề xuất công thức adaptive và chưa benchmark test.

## 1. Dữ liệu, phạm vi và kiểm chứng

Giữ nguyên MLP, labels, calibrated probabilities, hướng dự đoán, fixed τ=0,64 và phí 1 bps. Cùng **80.260 validation samples** (BTC 40.209; ETH 40.051). Fixed chọn 20.463 mẫu: coverage 25,50%, selective accuracy 68,93%, net 99,18 bps/mẫu chọn. Không đọc test predictions; không thay dataset, artifacts cũ hoặc source tác giả.

Feature tính từ **toàn bộ lịch sử mid-price unified theo từng symbol**, sau đó mới ghép validation một-một. Không tính rolling trên cohort đã lọc future return. Mỗi feature tại t chỉ dùng prices tới t, với quyết định diễn ra sau khi minute aggregate t đã quan sát xong. Window/lag là **số quan sát**, không phải phút lịch chính xác: 360 dòng median khoảng 361 phút. Không resample, forward-fill, winsorize, hoặc clip returns để làm đẹp feature. Lấy log-return từ mid-price hiện có, khác với mid_return clip của volatility feature gốc.

13 cấu hình feature cơ sở và 20 representations: 7 ACF signed, 7 |ACF| từ cùng tính toán, 3 VR, 3 ER. Các cấu hình chốt trước xem [feature_protocol.json](research_autocorrelation_analysis/validation_20261103/feature_protocol.json). Hai interaction diagnostics thêm vào mô hình thống kê, không phải feature search rộng hoặc adaptive methods mới. Không chọn window/lag dựa vào test.

**Kiểm chứng:** prefix-invariance PASS cho mọi cấu hình ở hai coin (bỏ phần tương lai không đổi feature quá khứ); direct finite-window ACF/VR trên các cửa sổ giá thật PASS; ER_360 trùng |signed efficiency_360| của bước regime; không missing trên toàn validation. Hash các inputs và source trước/sau phân tích không đổi. Xem [exploration_status.json](research_autocorrelation_analysis/validation_20261103/exploration_status.json), [script](research_autocorrelation_analysis/explore.py).

Đây vẫn là oracle-filtered cohort |future return|>1%, không có Neutral; label horizon 600 dòng, overlap và các giới hạn cleaning/split/calibration gốc vẫn tồn tại. Causal trailing calculation trên stream đã preprocess không chứng minh raw pipeline hoàn toàn causal. Labels chỉ dùng đánh giá/fit các statistical controls trong validation, không dùng tính ACF/VR/ER.

## 2. Features và công thức

Đặt x_t=log(P_t), r_t=x_t−x_{t−1}, với P_t là mid-price theo từng symbol.

### 2.1. Rolling ACF có dấu và trị tuyệt đối

```math
\bar r_{t,W}=\frac{1}{W}\sum_{j=0}^{W-1}r_{t-j},\qquad
\widehat\rho_{t,W}(k)=
\frac{\sum_{j=0}^{W-k-1}(r_{t-j}-\bar r_{t,W})(r_{t-j-k}-\bar r_{t,W})}
{\sum_{j=0}^{W-1}(r_{t-j}-\bar r_{t,W})^{2}}.
```


Cách đọc: tích các returns cách nhau k dòng, trừ **cùng trung bình của W returns**, chia tổng bình phương returns đã trừ trung bình. Đây là finite-sample ACF dạng không điều chỉnh n−k, không phải Pearson rolling correlation dùng hai mean riêng. Kết quả denominator zero là missing. Cấu hình: k=1,5 với W=60,360,1440; thêm k=15,W=360. Phân tích cả ρ và |ρ|, không xem chúng là hai phép đo độc lập.

ρ>0 mô tả cùng chiều ở lag k, ρ<0 mô tả đảo chiều ở lag đó; |ρ| chỉ cường độ phụ thuộc tuyến tính và bỏ dấu. Mô tả ba band: negative <−0,05; near-zero trong [−0,05;0,05]; positive >0,05. Band 0,05 chỉ để mô tả, **không phải ngưỡng significance**, không đổi theo performance. Dependence ở lag ngắn không tự chứng minh dự đoán được return horizon 600 dòng.

Tham khảo quy ước adjusted=False và các giới hạn inference trong [tài liệu ACF chính thức của statsmodels](https://www.statsmodels.org/v0.13.5/generated/statsmodels.tsa.stattools.acf.html). Phân tích này tự tính theo công thức trên, không cài hoặc chạy statsmodels tests.

### 2.2. Rolling Variance Ratio mô tả

```math
R^{(q)}_s=\sum_{i=0}^{q-1}r_{s-i},\qquad
\widehat{VR}_{t,W}(q)=
\frac{\mathrm{Var}_{ddof=1}\{R_s^{(q)}:s=t-W+q,\ldots,t\}}
{q\mathrm{Var}_{ddof=1}\{r_s:s=t-W+1,\ldots,t\}}.
```


Cách đọc: variance của tất cả W−q+1 overlapping q-return sums nằm **hoàn toàn trong cùng W returns**, chia q lần variance một-step. Cấu hình (W,q)=(360,5),(360,15),(1440,5). VR>1 phù hợp variance tăng nhanh hơn phép cộng variance không tương quan; VR<1 theo chiều ngược lại; VR≈1 gần tỷ lệ random-walk reference. Có finite-sample bias, overlap và heteroskedasticity; không coi lệch 1 là kiểm định bác bỏ random walk hoặc bằng chứng thị trường không hiệu quả.

[arch VarianceRatio documentation](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.VarianceRatio.html) phân biệt overlap, debiasing và robust inference. **Không chạy formal Lo–MacKinlay test**, không áp debiasing/robust z-stat của arch ở đây; đây là feature mô tả được định nghĩa riêng. VR và ACF có liên hệ cấu trúc qua variance của tổng returns, nên không coi hai nhóm evidence là độc lập.

### 2.3. Efficiency Ratio của log-price path

```math
ER_{t,W}=\frac{|x_t-x_{t-W}|}{\sum_{j=0}^{W-1}|r_{t-j}|},\qquad W\in\{60,360,1440\}.
```


Cách đọc: độ lớn dịch chuyển ròng chia tổng quãng đường log-price. ER trong [0,1], gần 1 khi đường đi cùng hướng hơn; gần 0 khi nhiều chuyển động bù nhau. **Không phải efficiency theo nghĩa efficient market hypothesis**, không có dấu và không mặc định ER cao thì MLP đáng tin hơn. Đây là log-price ER; không trình bày như một phiên bản dùng absolute dollar-price changes. ER_360 bằng trị tuyệt đối của efficiency feature đã nghiên cứu ở bước regime, không phải một nguồn dữ liệu mới độc lập.

## 3. Distribution, missing và outliers

Bảng trình bày bốn representations chính; missing bằng 0 ở cả 20 representations. `tukey_outlier_share` là tỷ lệ ngoài Q25−1,5IQR/Q75+1,5IQR, không phải tỷ lệ dữ liệu sai. Giữ mọi giá trị. [distributions.csv](research_autocorrelation_analysis/validation_20261103/distributions.csv) có p01/p99 và các window/lag còn lại.

'''+table(dist)+r'''

ACF lag-1,W=360 median gần zero (+0,0093 BTC; +0,0102 ETH), nhưng phân phối có cả hai dấu. VR(5,360) median 0,9554/0,9601; không suy ra mean reversion có lợi nhuận. ER_360 median chỉ 0,0465/0,0478, lý giải tại sao threshold ER>0,20 của bước trước tạo nhóm hiếm.

![Feature distributions](research_autocorrelation_analysis/validation_20261103/feature_distributions.png)

## 4. Quantiles và fixed-threshold performance trên toàn validation

Q1–Q5 tính riêng từng symbol bằng validation quantiles, chỉ phục vụ mô tả; giữ ties cùng nhóm, không dùng test quantiles. Các cột accuracy/coverage/selective accuracy/ECE là %, confidence_mean là probability. Mẫu số accuracy/ECE là toàn nhóm; selective metrics chỉ trên mẫu confidence≥0,64. Net và expected net là bps; expected net = coverage × net/mẫu chọn. Có đầy đủ counts, confidence min/median/P90/P95/max và mọi cấu hình trong [quantile_metrics.csv](research_autocorrelation_analysis/validation_20261103/quantile_metrics.csv).

'''+table(q)+r'''

Ví dụ: ETH |ACF_1_360| Q1 net −118,07 bps/mẫu chọn, Q5 +192,47 bps; BTC tương ứng +106,43 và +207,67. Nhưng raw accuracy ETH không tăng đơn điệu, và quan hệ có thể do các episode, direction hoặc volatility. Với signed ACF, BTC Q5 có **accuracy toàn nhóm 45,44%** nhưng selective accuracy 86,79%; selection thay đổi cohort rất mạnh. Không dùng accuracy toàn nhóm thay selective accuracy.

### Dấu ACF lag-1,W=360

'''+table(signed)+r'''

### Predicted Up/Down: Q1 và Q5 signed ACF lag-1,W=360

direction=0 là dự đoán Down, direction=1 là Up. Quantiles vẫn dùng cùng ranh giới theo asset, không chia lại theo direction.

'''+table(dr)+r'''

ETH Q1 Down có accuracy 97,16% nhưng chỉ **141 selections**, dưới cờ 200; không coi đó là 141 thử nghiệm độc lập. ETH Q5 Down 33,33%/−30,81 bps, trong khi Up 66,85%/+104,86 bps. Raw dependence theo direction rõ nhưng chưa đủ chứng minh một interaction ổn định.

![Quantile performance](research_autocorrelation_analysis/validation_20261103/quantile_performance.png)

## 5. Confidence reliability

ECE tính 10 bins cố định đều trên [0,1], weighted absolute difference giữa correctness và top-label confidence; không phải calibration riêng probability Up. Ví dụ signed ACF_1_360 BTC Q5 confidence trung bình 0,5782 nhưng accuracy toàn nhóm 45,44%, ECE 19,20%; ETH Q1 confidence 0,5556, accuracy 68,66%, ECE 13,16%. Confidence không đồng nghĩa cùng reliability ở mọi dependence state.

Calibration gốc đã fit trên validation; các số này không phải independent calibration assessment. Diagram chỉ vẽ bins n≥100; dữ liệu tất cả bins giữ trong [reliability_bins.csv](research_autocorrelation_analysis/validation_20261103/reliability_bins.csv).

![Reliability by ACF](research_autocorrelation_analysis/validation_20261103/reliability_by_acf.png)

## 6. Incremental information: trước/sau thêm feature

### 6.1. Statistical comparator và temporal protocol

Fit **mô hình thống kê chẩn đoán**, không train lại MLP và không dùng output này để giao dịch. M0 correctness: logistic L2 C=1; M0 return: ridge alpha=10, dự đoán signed sample net bps. Controls: confidence logit; exact symbol×confidence×predicted-direction strata; symbol; predicted direction; confidence×symbol/direction; log volatility_30m và log volatility_15d; tanh(trend_360/2), |tanh(trend)|; Bull/Bear indicators; linear calendar time. Strata ít hơn 20 train rows gộp OTHER; unseen evaluation strata cũng map OTHER. Không dùng validation labels để fit preprocessing của một fold tương lai.

M1 thêm từng representation; thêm hai diagnostics riêng cho ACF_1_360: (i) main + ACF×confidence-logit + ACF×predicted-direction; (ii) main + ACF×ETH. Không stack nhiều features để tìm combination thắng. Numeric scalers và categorical encoder fit **chỉ trên train prefix của fold**; C/alpha không tune. Logistic convergence warnings được coi lỗi; tất cả fits hoàn tất. Parameters/models lưu trong thư mục `statistical_controls/`, không thay classifier MLP gốc.

Chia validation thành bốn khoảng lịch bằng nhau theo ranh giới chung BTC/ETH; ba expanding folds: train quá khứ, evaluate khoảng kế tiếp. Purge train khi future label endpoint vượt eval-start, purge eval khi endpoint vượt eval-end; endpoint phục hồi bằng shift −600 trên unified timestamp, không thay endpoint bằng 600 phút lịch. Không bỏ bớt mẫu vì feature khó hoặc vì return xấu. Common finite cohort vẫn đủ 80.260 mẫu; 62.696 rows thuộc ba evaluation folds sau purge.

'''+table(folders.round(5))+r'''

Đây là forward validation của statistical comparators, **không phải OOS độc lập của MLP**: MLP/calibrator và exploratory selection đã dùng validation. Không đưa test predictions vào bất kỳ fold nào.

### 6.2. Health check: comparator không tổng quát hóa tốt

'''+table(health[['fold','n','correctness','baseline_p_mean','baseline_fraction_over95','baseline_logloss','confidence_reference_logloss','baseline_brier','confidence_reference_brier']].round(5))+r'''

Fold 1 actual correctness 38,52% nhưng M0 dự báo trung bình 93,94%, log loss 2,9515, tệ hơn confidence reference 0,7115. Hai folds sau M0 cũng tệ hơn reference. Temporal extrapolation, calendar/daily feature confounding và overfit quan hệ prefix là các **khả năng**, chưa ablate để xác định nguyên nhân chính xác. Không sửa hyperparameters hoặc bỏ time control sau khi thấy kết quả để tạo comparator thuận lợi hơn.

**Hệ quả:** gain khi thêm feature chỉ là relative gain so với comparator yếu này; không đủ để chứng minh feature tạo reliability model tốt hoặc trading edge. Confidence reference cũng đã được calibrated bằng toàn validation, nên không coi nó là benchmark calibration độc lập. Dù các models hội tụ số học, statistical validity/generalisability vẫn hạn chế.

### 6.3. Incremental metrics và effect sizes

Gain logloss/Brier/MAE/RMSE = M0 loss − M1 loss, dương là tốt hơn. Summary weighted theo số evaluation rows từng fold; riêng RMSE là **weighted mean của fold-RMSE reductions**, không gọi là pooled global RMSE. `effect_correct_pp`: average thay đổi xác suất correctness dự báo khi candidate từ train Q25 sang Q75, giữ controls khác cố định; `effect_net_bps` tương tự cho ridge. Đây là model-implied conditional contrast, không causal effect. Với interaction, candidate và các tích tương ứng cùng được thay; IQR fit từ train, không từ evaluation. Coefficient/odds ratio có trong CSV; không báo p-value IID.

'''+table(main_summary)+r'''

**Điểm đáng kiểm chứng, chưa là winner:** ACF lag-5,W=1440 có weighted Δlogloss +0,08932 và fold-IQR correctness contrast dương cả 3 folds. Nhưng weighted ΔBrier **−0,00509**, fold 2 mất logloss/RMSE, gain trung bình bị fold 1 với comparator rất yếu chi phối. |ACF lag-1,W=1440| có gain nhỏ hơn nhưng Brier và RMSE average dương; tác động đổi dấu giữa folds. ER_60 tăng logloss nhỏ ở cả 3 folds (+0,00252 average), return RMSE gain chỉ +0,043 bps; chưa có ý nghĩa kinh tế rõ.

Các candidates này dưới đây được trình bày để kiểm tra stability chứ không khóa method/window thắng:

'''+table(effect)+r'''

ACF_1_360 signed có weighted Δlogloss −0,02955, VR_5_360 −0,03165. Raw group differences không chuyển thành temporal prediction gain ổn định. Dự đoán signed return tốt hơn cũng không đồng nghĩa expected trading utility đã tăng, vì không có selection rule mới được triển khai.

![Incremental effects](research_autocorrelation_analysis/validation_20261103/incremental_effects.png)

### 6.4. Conditional-strata robustness, thêm ngày

Kiểm tra riêng từng asset bằng cách demean feature và outcomes trong exact confidence×predicted-direction×volatility-quintile×trend-quintile strata, sau đó partial out log micro/daily vol và continuous trend còn dư; thêm day strata cho spec thứ hai. Các quantiles chỉ dùng mô tả toàn validation. Không fit/đổi probabilities MLP. Singleton strata có residual zero, không tạo thông tin giả; số rows trong repeated strata ghi trong CSV. Daily log-vol hằng trong ngày bị bỏ khi không có variation còn lại.

Effect ghi theo thay đổi một **full-validation IQR** của feature, khác train-IQR counterfactual của temporal logistic model. Không so hai effect size như cùng một estimator. Không nhân quả và chưa có confidence interval hiệu chỉnh time dependence.

'''+table(cond)+r'''

ACF_5_1440 trước thêm day có conditional correctness +11,89 điểm phần trăm BTC/+6,93 ETH mỗi IQR; sau day thành +3,28 BTC/**−8,58 ETH**. Net conditional sau day chỉ +1,04 bps BTC/−25,84 ETH. |ACF_1_360| sau day −0,85 điểm phần trăm BTC nhưng +3,56 ETH. **Thông tin bổ sung phụ thuộc đặc tả và symbol; chưa có dấu chung đáng tin.**

### 6.5. Interactions

Joint ACF_1_360×confidence/direction diagnostic làm logloss tệ hơn ở **cả ba folds** (weighted gain −0,10158). Model có thể cho IQR correctness effect rất lớn ở predicted Up nhưng forecast loss xấu: đây là ví dụ effect size model-implied lớn không đồng nghĩa generalization tốt. Hệ số raw ACF×confidence đổi dấu từ khoảng −33,18 ở fold 1 sang +13,33/+7,20; scale là log-odds trên tích ACF×logit(confidence), không phải percentage points. Hệ số interaction raw và scalers lưu trong statistical_coefficients.json.

'''+table(interactions)+r'''

Asset-interaction variant cũng có weighted gain logloss âm (−0,02206). Đây chưa phải evidence rằng interaction không tồn tại; chỉ cho thấy đặc tả nhỏ đã kiểm tra không tạo improvement ổn định ở các periods này. Không thêm nhiều interactions hậu nghiệm để tìm winner.

## 7. Volatility, trend, stability và sensitivity

- Spearman ACF_1_360 với micro volatility chỉ −0,045 BTC/−0,090 ETH; với trend_360 +0,143/+0,173. |ACF_1_360| với micro volatility khoảng −0,004/+0,006, nhưng ETH với daily volatility −0,179. Correlation thấp không chứng minh conditional independence.
- ER_360 với micro volatility +0,311/+0,278, còn signed trend −0,040/−0,075. ER vốn lấy absolute signed efficiency của bước regime, có thể trùng thông tin về trend/noise và volatility; không coi đây là nhóm evidence độc lập.
- ACF_5_1440 với micro volatility −0,078/−0,054 và signed trend khoảng +0,017/+0,011; vẫn có effect đổi dấu sau day controls. Không thể quy nó hoàn toàn cho volatility, cũng chưa chứng minh phụ thuộc thời gian thật là nguyên nhân.
- Window/lag sensitivity lớn: lag-5 1.440 dòng có relative logloss gain tốt hơn lag-1 cùng window, trong khi ACF_1_1440 weighted gain −0,15447. Đây là exploratory multiple comparisons; không chọn method chỉ vì một cấu hình đứng đầu 22 diagnostics.
- Weekly stability không đồng nhất. ETH |ACF_1_360| Q1 có tuần net khoảng −1.337,37 bps/mẫu chọn và tuần +207,13 bps (chỉ tính các tuần ≥200 selections). Signed ACF_1_360 ETH Q5 dao động −407,32 tới +241,36 bps. BTC signed Q1 −125,21 tới +253,91 bps. Mức gain có thể tập trung ở một số episode; không loại tuần xấu.
- |ACF| Q5 ETH chỉ có hai tuần đạt ≥200 selections dù tổng nhóm có 1.801 selections; nhiều row không đồng nghĩa nhiều episode độc lập. Các ô Up/Down hiếm được gắn few_selected/few_samples, không dùng cực trị làm conclusion.
- Sliding windows, labels 600 dòng overlap và autocorrelation của chính feature/outcomes làm số mẫu hiệu dụng nhỏ hơn count. Chưa tính HAC/block-bootstrap CIs; ba folds không đủ để khẳng định significance. Không báo IID p-values. Tất cả values/outliers vẫn giữ nguyên; Tukey flag chỉ mô tả, không sửa data.

![Weekly stability](research_autocorrelation_analysis/validation_20261103/weekly_stability.png)

## 8. Trả lời câu hỏi và quyết định nghiên cứu

1. **Feature quan hệ mạnh nhất với correctness?** Không có winner ổn định. ACF_5_1440 có association/relative forward-model gain nổi bật nhưng Brier giảm và ETH conditional effect đảo dấu sau day control. |ACF_1_1440| có gain nhỏ hơn, vẫn không ổn định hoàn toàn. Không chọn chỉ dựa raw selective accuracy.
2. **Feature liên hệ expected trading utility?** Raw |ACF_1_360| là giả thuyết đáng chú ý ở ETH: expected net Q1 −36,55 bps/mẫu đủ điều kiện, Q5 +43,28; BTC +26,51/+46,64. Nhưng after-controls sign khác giữa asset và không có temporal expected-utility improvement của strategy mới nào được đo. Ridge ΔRMSE là prediction loss, không phải utility gain.
3. **Thông tin ngoài confidence/volatility/regime/symbol/direction?** Có residual associations trong các controls đã thử, chưa có bằng chứng vững về incremental predictive value có thể tổng quát hóa. Comparator health yếu, effects nhạy day controls, nên không tuyên bố independence hoặc causal dependence effect.
4. **Ổn định thời gian và asset?** Chưa. Folds, tuần, direction và hai coin khác nhau; một số dấu đảo. ER_60 logloss gains nhất quán pooled nhưng nhỏ, không đi kèm return gain rõ và effect sign cũng không cố định.
5. **Signed ACF, |ACF|, VR, ER hay kết hợp?** Giữ signed ACF và |ACF| như hai hypotheses phân biệt; chưa chọn combination. VR không cho relative gain ổn định; ER chủ yếu là kiểm tra trend/noise với incremental effect nhỏ. Kết hợp nhiều nhóm lúc này dễ nhân multiple-testing và thông tin trùng lặp.
6. **Đủ cơ sở xây adaptive threshold?** Chưa đủ để khẳng định nên đổi threshold hoặc có edge. CONDITIONAL GO chỉ cho kiểm chứng tiếp dependence–reliability với independent periods và comparator ổn định; chưa triển khai method hoặc chọn tham số để mang sang test.
7. **Ba hypotheses:** H1: |ACF| liên hệ với reliability/tail-loss theo asset và direction hơn là một quy tắc momentum/mean-reversion chung. H2: long-window lag-5 association phần lớn có thể do calendar episodes; nó sẽ yếu hoặc đảo khi kiểm tra trong ngày/period độc lập. H3: VR/ER raw utility differences chủ yếu phản ánh trend, volatility và oracle-cohort composition, không chắc còn incremental value sau kiểm soát. Chưa thiết kế công thức adaptive để kiểm chứng các hypotheses này.

## 9. Artifacts và tái lập

- [Feature/distribution/quantile metrics](research_autocorrelation_analysis/validation_20261103/quantile_metrics.csv), [direction breakdown](research_autocorrelation_analysis/validation_20261103/direction_quantiles.csv), [weekly](research_autocorrelation_analysis/validation_20261103/weekly_quantiles.csv), [correlations](research_autocorrelation_analysis/validation_20261103/control_correlations.csv).
- [Temporal incremental metrics](research_autocorrelation_analysis/validation_20261103/temporal_incremental_metrics.csv), [summary](research_autocorrelation_analysis/validation_20261103/incremental_summary.csv), [conditional strata](research_autocorrelation_analysis/validation_20261103/conditional_strata_effects.csv), [comparator health](research_autocorrelation_analysis/validation_20261103/control_model_health.csv).
- validation_features.parquet giữ hàng/cột feature và labels gốc; statistical_validation_predictions.parquet chỉ giữ predictions của **statistical validation comparators**, không thay MLP predictions. Fold models/scalers trong statistical_controls/, protocol/coefficients/status JSON đầy đủ. MLP, dataset, original source và artifacts Volatility/Regime methods cũ giữ nguyên. [Integrity checks](research_autocorrelation_analysis/validation_20261103/integrity_checks.json) đối chiếu hash classifier gốc, validation predictions và source với metadata lần Regime-B: PASS; không đọc test predictions để làm kiểm tra này.
- Chạy lần lượt `reproduction_setup/venv/Scripts/python.exe research_autocorrelation_analysis/explore.py`, `conditional_analysis.py`, `summarize.py`, `write_report.py` (ba lệnh sau cùng đường dẫn research_autocorrelation_analysis/). Hai script đầu từ chối overwrite output/stats-model directory hiện tại. Để tái lập, đổi OUT sang thư mục mới trong module explore; không xóa kết quả cũ. Không cài thư viện.

'''+f"Runtime exploration {statuses[0]['elapsed_sec']:.2f}s, conditional comparators {statuses[1]['elapsed_sec']:.2f}s; chưa gồm các bước tóm tắt/báo cáo. Peak working set lần lượt {statuses[0]['memory']['peak_ws_gib']:.3f}/{statuses[1]['memory']['peak_ws_gib']:.3f} GiB.\n"
    import re
    text=re.sub(r'(!?\[[^\]]*\]\()([^)]+)(\))', lambda m: m.group(1)+(m.group(2) if m.group(2).startswith(('http://','https://','mailto:','#')) else '../'+m.group(2))+m.group(3), text)
    (ROOT/'analysis/AUTOCORRELATION_EXPLORATORY_ANALYSIS.md').write_text(text,encoding='utf-8')
    print('Wrote AUTOCORRELATION_EXPLORATORY_ANALYSIS.md')

if __name__=='__main__':main()

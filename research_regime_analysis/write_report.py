"""Render Vietnamese report from saved validation-only statistics."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research_regime_analysis/validation_20261103'

def table(x):
    return '| '+' | '.join(map(str,x.columns))+' |\n|'+'|'.join(['---']*len(x.columns))+'|\n'+'\n'.join('| '+' | '.join('undefined' if pd.isna(v) else str(v) for v in row)+' |' for row in x.itertuples(index=False,name=None))

def main():
    allm=pd.read_csv(P/'regime_metrics.csv')
    q=allm[['definition','symbol','regime','n','regime_share','up_n','down_n','accuracy','selected_n','coverage','selective_accuracy','net_selected_bps']].copy()
    for c in ['regime_share','accuracy','coverage','selective_accuracy']:q[c]=(q[c]*100).round(2)
    q.net_selected_bps=q.net_selected_bps.round(2)
    primary=allm[allm.definition=='trend_360']
    cal=primary[['symbol','regime','confidence_mean','confidence_median','confidence_p90','calibration_gap','ece_10','selected_days']].copy()
    for c in ['confidence_mean','confidence_median','confidence_p90']:cal[c]=cal[c].round(4)
    for c in ['calibration_gap','ece_10']:cal[c]=(cal[c]*100).round(2)
    direction=pd.read_csv(P/'direction_metrics.csv')
    d=direction[(direction.definition=='trend_360')&direction.condition.str.startswith('predicted')][['symbol','regime','condition','n','accuracy','selected_n','selective_accuracy','net_selected_bps']].copy()
    for c in ['accuracy','selective_accuracy']:d[c]=(d[c]*100).round(2)
    d.net_selected_bps=d.net_selected_bps.round(2)
    recall=direction[(direction.definition=='trend_360')&direction.condition.str.startswith('actual')][['symbol','regime','condition','n','accuracy']].copy()
    recall.accuracy=(recall.accuracy*100).round(2)
    effects=pd.read_csv(P/'controlled_effects.csv')
    e=effects[(effects.definition=='trend_360')&(effects.outcome=='correct')].pivot(index=['symbol','contrast'],columns='controls',values='effect').reset_index()
    for c in ['confidence','confidence_vol','confidence_vol_day']:e[c]=(100*e[c]).round(2)
    robust=pd.read_csv(P/'continuous_volatility_controls.csv')
    rb=robust[(robust.definition=='trend_360')&(robust.outcome=='correct')].pivot(index=['symbol','contrast'],columns='controls',values='effect').reset_index()
    for c in ['confidence_continuous_vol','confidence_continuous_vol_day']:rb[c]=(100*rb[c]).round(2)
    e=e.merge(rb,on=['symbol','contrast'])
    correlations=pd.read_csv(P/'volatility_correlations.csv').round(3)
    switches=pd.read_csv(P/'calendar_switching.csv')[['definition','symbol','switches','switch_rate','median_run_rows','p90_run_rows']].copy()
    switches.switch_rate=(100*switches.switch_rate).round(2)
    switches.p90_run_rows=switches.p90_run_rows.round(1)
    elapsed=pd.read_csv(P/'lookback_elapsed.csv').round(2)
    meta=json.loads((P/'metadata.json').read_text())
    text=r'''# Market Regime Exploratory Analysis — validation BTC/ETH

Ngày 03/11/2026. **Chỉ phân tích dữ liệu validation. Chưa đề xuất, fit hay triển khai regime-adaptive threshold; không benchmark test.** Fixed τ=0,64, MLP, probabilities, labels và phí 1 bps giữ nguyên. Các hồi quy trong báo cáo là công cụ thống kê mô tả correctness/return, không thay predictive model.

## 1. Phạm vi, kiểm tra và giới hạn suy luận

Toàn bộ validation gốc **80.260 mẫu** (BTC 40.209; ETH 40.051), ghép một-một `(symbol,timestamp)` với unified. BTC 23/06–15/08/2024 UTC, ETH 26/06–17/08/2024 UTC. Không chia nhỏ tập để thay đổi kết quả baseline. Cả 5 định nghĩa regime không có missing trên cohort này. Baseline xác minh lại: **20.463 mẫu chọn, coverage 25,50%, selective accuracy 68,93%, net 99,18 bps/mẫu chọn**.

Cohort đã oracle-filter theo |future return|>1%; horizon 600 dòng, không chắc 600 phút. Không có Neutral trong các mẫu đang phân tích. Tỷ trọng regime/labels trong báo cáo là trên cohort đã lọc này, không phải phân phối mọi thời điểm thị trường. MLP và calibration đã sử dụng validation; toàn validation từng được xem, nên đây là retrospective/exploratory. Các hàng, labels và returns chồng lấn: số selections không phải vị thế độc lập. Không cộng returns thành portfolio PnL hoặc annualized Sharpe.

Regime được xây từ giá/feature quá khứ trên **toàn bộ lịch sử unified**, rồi mới ghép validation; không tính rolling trên cohort oracle-filtered. Prefix-invariance của trend 60/360/1.440 dòng: PASS cho cả coin. Đây là causal construction theo dòng dữ liệu đã preprocess; không khắc phục các giới hạn cleaning/timestamp/availability từ upstream. Khi nói thông tin ở t, quy ước quyết định diễn ra sau khi minute aggregate tại t đã quan sát xong. Không dùng future label/return để định nghĩa regime.

## 2. Định nghĩa nhỏ được chốt trước phân tích kết quả

Các feature repo hiện có gồm `ma_5d/10d/20d`, `momentum_1d/5d/20d`, `returns`, `rsi_14`, daily/micro volatility; `volatility_regime` là cờ volatility, không phải Bull/Bear trend. Repo MA dùng Close.shift(1).rolling; momentum dùng Close.pct_change(lag).shift(1). Xem [macro features](data_preprocessing.py#L1065), [momentum](data_preprocessing.py#L1102), [macro date +1 ngày](data_preprocessing.py#L1852).

Gọi P_t là mid-price, r_t=log(P_t/P_{t−1}); t và lookback micro là **chỉ số dòng theo symbol**. Không clip r khi tính chỉ số trend mới; volatility_30m gốc dùng công thức clip của repo và được giữ nguyên. Quy tắc chung: score > a là Bull, score < −a là Bear, còn lại Sideways; giá trị đúng ranh giới thuộc Sideways. Các a là hằng số phân tích chốt trước, không tối ưu theo performance.

### A. Trend chuẩn hóa theo độ biến động đường giá

```math
T_{t,W}=\frac{\log P_t-\log P_{t-W}}{\sqrt{\sum_{j=0}^{W-1}r_{t-j}^{2}}},\qquad W\in\{60,360,1440\},\quad a=1.
```


Cách đọc: log-return quá khứ chia căn tổng bình phương log-return của cùng cửa sổ. Tách hướng và độ mạnh tương đối; không coi score này là thống kê kiểm định chuẩn do returns phụ thuộc. Ưu: causal, scale-free và so sánh hai coin thuận tiện. Nhược: denominator phụ thuộc volatility, nhạy cửa sổ và có thể đảo nhãn quanh ranh giới. 60/360/1.440 dòng gần 1 giờ/6 giờ/1 ngày, **không phải cửa sổ clock-time chính xác**. Đây là một họ định nghĩa với ba kiểm tra sensitivity, không chọn window bằng test.

### B. Directional efficiency, W=360

```math
E_{t,360}=\frac{\log P_t-\log P_{t-360}}{\sum_{j=0}^{359}|r_{t-j}|},\qquad a=0.20.
```


Cách đọc: dịch chuyển ròng có dấu chia tổng chiều dài đường giá. Trong [−1,1]; |E| cao thể hiện đường đi cùng hướng hơn, không chứng minh thị trường hiệu quả theo nghĩa kinh tế học. Ưu: dễ giải thích và bounded. Nhược: với dữ liệu nhiễu 360 dòng, ngưỡng 0,20 có thể làm trend regime cực hiếm; kết quả thực tế xác nhận điều đó. Không hạ ngưỡng hậu nghiệm để tạo nhóm đẹp hơn.

### C. Daily MA gap 5/20

```math
G_t=\frac{MA^{daily}_{5}(t)}{MA^{daily}_{20}(t)}-1,\qquad a=0.005.
```


Cách đọc: MA 5 ngày cao hơn MA 20 ngày trên 0,5% là Bull, thấp hơn trên 0,5% là Bear. Dùng hai MA đã có, trên daily Close từ macro dataset, rolling lag một daily row và join ngày D vào D+1. Ưu: ít switching, không cần thêm nguồn OHLCV. Nhược: trễ, rất ít trạng thái độc lập trong validation khoảng hai tháng, mất cân bằng; phụ thuộc chất lượng daily source sẵn có.

**Price slope:** chưa thêm hồi quy slope hoặc indicator thứ sáu. T_W đã đo hướng dịch chuyển/độ mạnh; thử thêm slope lúc này tăng số lựa chọn hậu nghiệm. RSI/momentum cũng chưa thêm vào sweep này. Đây là lựa chọn phạm vi phân tích, không khẳng định các indicator đó kém.

### Thời gian lịch thực của lookback micro

'''+table(elapsed)+'''

Median 360 dòng là 361 phút cho cả hai coin; P95 BTC 370 phút, ETH 375 phút. Không tự điền missing minutes hoặc đổi giá để làm cửa sổ tròn giờ.

## 3. Kết quả đầy đủ theo định nghĩa, symbol và regime

Các cột share/accuracy/coverage/selective_accuracy là %. `up_n/down_n` là counts labels tương lai dùng **đánh giá**, không dùng xác định regime. Return là bps/mẫu chọn sau phí 1 bps. `n` là cohort size, `selected_n` là số vượt fixed threshold. Hàng regime không xuất hiện có count=0; đặc biệt BTC daily MA không có Sideways. Một mẫu được dự đoán Up/Down ở mọi regime; Bull không đồng nghĩa future label Up.

'''+table(q)+'''

Trend 360 dòng được dùng làm ví dụ trọng tâm để đọc bảng, không được tuyên bố là window tối ưu. BTC Bull chỉ có 43,10% future labels Up; ETH Bull 39,69%. Điều này cho thấy trend quá khứ và label tương lai là hai đại lượng khác nhau, đồng thời có thể chịu tác động oracle filtering/horizon và mean reversion; chưa thể xác định nguyên nhân.

ETH Bear trend_360 có selective accuracy 61,29% nhưng net **−93,87 bps/mẫu chọn**. Sai lầm có độ lớn return lớn hơn lợi nhuận khi đúng là một khả năng trực tiếp; accuracy không chứa thông tin về độ lớn thắng/thua. Không suy diễn rằng 61% correctness đủ để có edge.

## 4. Confidence và calibration theo regime

Confidence là max(p_up,p_down). Calibration đang xét **top-label correctness**, không phải reliability diagram của riêng probability Up. `calibration_gap` = accuracy toàn nhóm − confidence trung bình, đơn vị điểm phần trăm; ECE dùng 10 bin cố định đều trên [0,1], weighted absolute gap. ECE và confidence min/median/P90/P95/max có đủ cho cả năm định nghĩa trong CSV; bảng dưới trình bày trend_360.

'''+table(cal)+'''

ETH Bear có confidence trung bình 0,5747 nhưng accuracy toàn nhóm 64,49% (gap +7,02 điểm phần trăm); ECE 10,21 điểm phần trăm cho thấy sai lệch giữa các confidence bins có thể bù nhau ở mean gap. BTC Sideways gap −1,99 điểm phần trăm. Cùng confidence không thể mặc định có cùng reliability giữa regime; tuy nhiên calibration fit ngay validation nên các con số này không phải chất lượng calibration trên dữ liệu độc lập. [Reliability diagram](research_regime_analysis/validation_20261103/reliability_by_regime.png) chỉ vẽ bin n≥100, bảng CSV giữ mọi bin.

## 5. Up có dễ dự đoán hơn Down không?

Phải phân biệt **precision của dự đoán Up/Down** và **recall theo label thật**. Bảng đầu nhóm theo hướng MLP thực sự dự đoán; accuracy ở đây là directional precision. Bảng sau nhóm theo label thật; accuracy toàn nhóm là class recall. Không đọc selective class recall thành recall toàn bộ cơ hội vì tập mẫu số sau threshold đã đổi.

'''+table(d)+'''

Class recall trước threshold:

'''+table(recall)+'''

Không có quy luật Up luôn tốt hơn Down. Ví dụ ETH Bull: precision trước threshold Down 69,18%, Up 53,45%; sau threshold Down 99,38% trên 324 selections, Up 58,00% trên 1.612 selections. ETH Bear: Up sau threshold 62,18% trên 1.621 selections nhưng net −106,06 bps; Down 54,00% trên 200 selections, net +4,94 bps. Các số cực đoan có thể do cụm thời gian và biến động lớn, không phải hàng trăm thử nghiệm độc lập. Label Down được recall cao hơn Up trước threshold ở cả ba regime của hai coin; MLP cũng dự đoán Down nhiều hơn, nên không nhầm precision với khả năng nhận diện toàn bộ Up events.

## 6. Thông tin bổ sung sau confidence, symbol và volatility

### Phương pháp thống kê

Hồi quy tuyến tính mô tả với fixed effects của confidence chính xác, làm riêng từng symbol. Dùng hai dummy Bull và Bear, Sideways làm mốc. Trừ trung bình các biến trong từng stratum rồi least squares; hệ số correctness là chênh lệch xác suất đúng điều chỉnh, đơn vị điểm phần trăm. Đây là linear probability association, không phải probability model dùng thực thi và không có diễn giải nhân quả. Không báo p-value IID.

Ba mức kiểm soát: confidence; confidence + quintile volatility_30m theo symbol; thêm ngày UTC. Quintile dùng toàn validation chỉ phục vụ mô tả. Robustness thứ hai dùng log(volatility_30m) và log(volatility_15d) liên tục, kèm confidence hoặc confidence+day. Daily volatility hằng theo symbol/day nên bị hấp thụ bởi day fixed effects, không fit thêm một hệ số daily độc lập. Tính log/control bằng float64 tránh rounding tạo ra variation giả trong daily covariate. Các tương quan/hiệu ứng vẫn có thể còn confounding phi tuyến, direction hoặc cấu trúc thời gian chưa mô hình hóa.

### Effect sizes trend_360: adjusted correctness, điểm phần trăm

'''+table(e)+'''

BTC Bear có +5,49 điểm phần trăm khi chỉ kiểm soát confidence, nhưng −7,91 khi thêm vol-quintile/day; với continuous vol/day là −9,90. ETH Bull giữ dấu dương qua các đặc tả, từ +7,00 đến +9,51 điểm phần trăm; ETH Bear vẫn dương nhưng mức thay đổi đáng kể. Đây là bằng chứng heterogeneity, **chưa phải tín hiệu độc lập ổn định đủ để giao dịch**.

Với continuous vol/day controls, adjusted net trên **toàn bộ cohort, chưa threshold** của Bull so với Sideways chỉ +4,60 bps BTC, +8,06 ETH; Bear −67,62 BTC và −23,90 ETH. Hệ số correctness dương không đảm bảo hiệu ứng return dương. Xem [conditional effects](research_regime_analysis/validation_20261103/controlled_effects.csv) và [continuous controls](research_regime_analysis/validation_20261103/continuous_volatility_controls.csv). Không dùng các hồi quy này để thay probability hoặc threshold.

### Tương quan regime score với volatility

'''+table(correlations)+'''

Trend_360 signed score có Spearman với volatility_30m −0,131 BTC, −0,174 ETH; |score| tương quan +0,309 và +0,265. Không phải đồng nhất hoàn toàn nhưng có liên hệ vừa phải với độ mạnh trend. Vì trend đã chuẩn hóa bằng realized quadratic variation, vừa có quan hệ cấu trúc vừa có quan hệ thực nghiệm; không coi việc chia denominator đã loại hết volatility. MA gap và volatility cũng khác nhau theo asset.

## 7. Stability, switching và sensitivity

Switching tính trên **mọi micro row trong khoảng lịch validation mỗi coin**, không chỉ những hàng oracle-filtered; tránh nhầm gap giữa các mẫu được chọn thành regime transition. Run length có hai run biên bị censor. Median run ngắn phản ánh nhiều lần flicker quanh ranh giới ngay cả khi có một số run dài; không đồng nghĩa thị trường thay regime kinh tế mỗi vài phút.

'''+table(switches)+'''

Trend_360 có switch rate khoảng 3%/row và median run 3–4 dòng; trend_60 khoảng 7%, trend_1440 khoảng 1,3–1,5%. Vì vậy nhãn rời rạc chưa ổn định quanh cutoff. Daily MA chỉ đổi 2 lần BTC/4 lần ETH trong toàn khoảng: hàng chục nghìn dòng không phải hàng chục nghìn episode regime độc lập.

**Nhóm ít mẫu:** efficiency_360 Bull chỉ BTC 16 mẫu/2 selections và ETH 19 mẫu/0 selections; Bear BTC 134 mẫu/118 selections trong 3 ngày, ETH 366 mẫu/305 selections trong 3 ngày. Không diễn giải các hệ số hiệu ứng rất lớn của nhóm này. Daily MA BTC không có Sideways nên hệ số Bull/Bear-vs-Sideways không nhận diện; ETH Sideways 1.522 mẫu nhưng chỉ 2 selections. Các ô undefined và rank-deficient được giữ nguyên, không chế ra accuracy/profit cho nhóm rỗng.

**Thời gian:** với trend_360, BTC Bear tuần 01–07/07 có 335 selections, accuracy 100%, net +390,24 bps; tuần 05–11/08 có 550 selections, accuracy 41,27%, net −21,21 bps. ETH Bear tuần 29/07–04/08 có 265 selections, accuracy 0%, net −1.547,43 bps; tuần 05–11/08 có 1.064 selections, accuracy 64,38%, net +107,15 bps. Đây là dấu hiệu kết quả tập trung theo episode; không chọn chỉ những tuần thuận lợi. Các tuần ít hơn 200 selections cũng được giữ trong CSV nhưng không dùng làm kết luận mạnh.

**Sensitivity:** BTC Bear selected accuracy ở W=60/360/1440 lần lượt 81,69%/76,24%/89,38%; ETH Bull 58,83%/64,93%/78,63%. Những thay đổi này không chứng minh window dài hơn tốt hơn: regime membership, số ngày và return distribution đã đổi. Hiệu ứng sau day controls cũng khác dấu/độ lớn theo window. Không chọn window thắng dựa vào metrics này; cần nghiên cứu kiểm chứng riêng.

## 8. Kết luận theo câu hỏi nghiên cứu

1. **Regime liên hệ rõ nhất với reliability?** Không có một regime thắng thống nhất hai coin. Trong nhóm đủ samples, trend_360 Bull của ETH giữ liên hệ correctness dương qua các đặc tả; trend_1440 có khác biệt raw mạnh nhưng sensitivity/day confounding lớn. Bull/Bear của BTC không ổn định. Không tuyên bố trend_360 tối ưu toàn cục.
2. **Có thêm thông tin ngoài confidence/symbol/volatility?** Có residual association trên validation, nhất là ETH Bull, nhưng effect size phụ thuộc controls và thời gian. Chưa chứng minh incremental predictive information ngoài mẫu, độc lập volatility hoặc causal trend effect.
3. **Có cơ sở nghiên cứu threshold theo regime?** Có cơ sở cho giả thuyết về reliability có điều kiện và direction, nhưng chưa có bằng chứng để đổi fixed threshold. Accuracy, calibration và net không cùng chiều; phải giữ nguyên kết luận khám phá, chưa đề xuất công thức adaptive.
4. **Discrete hay continuous?** Nên ưu tiên nghiên cứu chỉ số trend liên tục, đồng thời giữ Bull/Bear/Sideways để diễn giải. Discretization hiện có flicker ở micro và ô hiếm ở efficiency/daily MA; continuous tránh mất thông tin tại cutoff nhưng vẫn cần xử lý sensitivity và confounding. Đây là lựa chọn đối tượng nghiên cứu, không phải một threshold method mới.
5. **Ba hypotheses đáng kiểm chứng:** (H1) past trend liên hệ với reliability khác nhau theo asset và hướng dự đoán, không chỉ một hiệu ứng Bull/Bear chung; (H2) reliability theo trend vẫn có residual association sau volatility controls nhưng return effect yếu hơn, nhất là ETH Bull; (H3) một phần lớn raw regime effect là episode/time concentration và oracle-cohort composition, sẽ yếu đi khi kiểm tra trên periods độc lập và có đủ regime episodes. Chưa fit/triển khai method để kiểm chứng các hypotheses này.

## 9. Artifacts, biểu đồ và tái lập

![Performance theo trend 360](research_regime_analysis/validation_20261103/trend_360_performance.png)

![Reliability theo regime](research_regime_analysis/validation_20261103/reliability_by_regime.png)

![Tỷ trọng regime từng ngày](research_regime_analysis/validation_20261103/daily_regime_shares.png)
- [Regime metrics, đầy đủ confidence distribution](research_regime_analysis/validation_20261103/regime_metrics.csv), [direction metrics](research_regime_analysis/validation_20261103/direction_metrics.csv), [weekly metrics](research_regime_analysis/validation_20261103/weekly_metrics.csv).
- [Metadata/checksums](research_regime_analysis/validation_20261103/metadata.json): input hashes trước/sau không đổi, không đọc test predictions, không fit threshold.
- Script [analyze_validation.py](research_regime_analysis/analyze_validation.py) và [robustness.py](research_regime_analysis/robustness.py), [write_report.py](research_regime_analysis/write_report.py). Lệnh lần lượt: `reproduction_setup/venv/Scripts/python.exe research_regime_analysis/analyze_validation.py`, rồi tương tự với `robustness.py` và `write_report.py`. Script đầu từ chối overwrite output hiện tại; để tái lập vào thư mục mới cần đổi hằng OUT trước khi chạy, không xóa kết quả cũ.

Runtime phân tích chính: '''+str(round(meta['elapsed_sec'],2))+''' giây; chưa tính bước robustness và dựng báo cáo. Không cài thư viện, không train lại MLP, không thay source tác giả, dataset hay kết quả Fixed/A/B cũ.
'''
    import re
    text=re.sub(r'(!?\[[^\]]*\]\()([^)]+)(\))', lambda m: m.group(1)+(m.group(2) if m.group(2).startswith(('http://','https://','mailto:','#')) else '../'+m.group(2))+m.group(3), text)
    (ROOT/'analysis/REGIME_EXPLORATORY_ANALYSIS.md').write_text(text,encoding='utf-8')
    print('Wrote REGIME_EXPLORATORY_ANALYSIS.md')

if __name__=='__main__':main()

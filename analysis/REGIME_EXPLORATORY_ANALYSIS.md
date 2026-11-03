# Market Regime Exploratory Analysis — validation BTC/ETH

Ngày 03/11/2026. **Chỉ phân tích dữ liệu validation. Chưa đề xuất, fit hay triển khai regime-adaptive threshold; không benchmark test.** Fixed τ=0,64, MLP, probabilities, labels và phí 1 bps giữ nguyên. Các hồi quy trong báo cáo là công cụ thống kê mô tả correctness/return, không thay predictive model.

## 1. Phạm vi, kiểm tra và giới hạn suy luận

Toàn bộ validation gốc **80.260 mẫu** (BTC 40.209; ETH 40.051), ghép một-một `(symbol,timestamp)` với unified. BTC 23/06–15/08/2024 UTC, ETH 26/06–17/08/2024 UTC. Không chia nhỏ tập để thay đổi kết quả baseline. Cả 5 định nghĩa regime không có missing trên cohort này. Baseline xác minh lại: **20.463 mẫu chọn, coverage 25,50%, selective accuracy 68,93%, net 99,18 bps/mẫu chọn**.

Cohort đã oracle-filter theo |future return|>1%; horizon 600 dòng, không chắc 600 phút. Không có Neutral trong các mẫu đang phân tích. Tỷ trọng regime/labels trong báo cáo là trên cohort đã lọc này, không phải phân phối mọi thời điểm thị trường. MLP và calibration đã sử dụng validation; toàn validation từng được xem, nên đây là retrospective/exploratory. Các hàng, labels và returns chồng lấn: số selections không phải vị thế độc lập. Không cộng returns thành portfolio PnL hoặc annualized Sharpe.

Regime được xây từ giá/feature quá khứ trên **toàn bộ lịch sử unified**, rồi mới ghép validation; không tính rolling trên cohort oracle-filtered. Prefix-invariance của trend 60/360/1.440 dòng: PASS cho cả coin. Đây là causal construction theo dòng dữ liệu đã preprocess; không khắc phục các giới hạn cleaning/timestamp/availability từ upstream. Khi nói thông tin ở t, giả định quyết định diễn ra sau khi minute aggregate tại t đã quan sát xong. Không dùng future label/return để định nghĩa regime.

## 2. Định nghĩa nhỏ được chốt trước phân tích kết quả

Các feature repo hiện có gồm `ma_5d/10d/20d`, `momentum_1d/5d/20d`, `returns`, `rsi_14`, daily/micro volatility; `volatility_regime` là cờ volatility, không phải Bull/Bear trend. Repo MA dùng Close.shift(1).rolling; momentum dùng Close.pct_change(lag).shift(1). Xem [macro features](../data_preprocessing.py#L1065), [momentum](../data_preprocessing.py#L1102), [macro date +1 ngày](../data_preprocessing.py#L1852).

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

| symbol | window_rows | median_minutes | p95_minutes | max_minutes | share_over_1_5_window |
|---|---|---|---|---|---|
| BTC_USDT | 60 | 60.0 | 61.0 | 151.0 | 0.0 |
| BTC_USDT | 360 | 361.0 | 370.0 | 529.0 | 0.0 |
| BTC_USDT | 1440 | 1444.0 | 1476.0 | 1644.0 | 0.0 |
| ETH_USDT | 60 | 60.0 | 62.0 | 218.0 | 0.01 |
| ETH_USDT | 360 | 361.0 | 375.0 | 576.0 | 0.01 |
| ETH_USDT | 1440 | 1444.0 | 1524.0 | 1719.0 | 0.0 |

Median 360 dòng là 361 phút cho cả hai coin; P95 BTC 370 phút, ETH 375 phút. Không tự điền missing minutes hoặc đổi giá để làm cửa sổ tròn giờ.

## 3. Kết quả đầy đủ theo định nghĩa, symbol và regime

Các cột share/accuracy/coverage/selective_accuracy là %. `up_n/down_n` là counts labels tương lai dùng **đánh giá**, không dùng xác định regime. Return là bps/mẫu chọn sau phí 1 bps. `n` là cohort size, `selected_n` là số vượt fixed threshold. Hàng regime không xuất hiện có count=0; đặc biệt BTC daily MA không có Sideways. Một mẫu được dự đoán Up/Down ở mọi regime; Bull không đồng nghĩa future label Up.

| definition | symbol | regime | n | regime_share | up_n | down_n | accuracy | selected_n | coverage | selective_accuracy | net_selected_bps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trend_60 | BTC_USDT | Bear | 5493 | 13.66 | 2831 | 2662 | 57.07 | 1218 | 22.17 | 81.69 | 224.23 |
| trend_60 | BTC_USDT | Bull | 5282 | 13.14 | 2449 | 2833 | 59.41 | 1258 | 23.82 | 75.83 | 173.64 |
| trend_60 | BTC_USDT | Sideways | 29434 | 73.2 | 14878 | 14556 | 57.04 | 7394 | 25.12 | 77.05 | 173.6 |
| trend_60 | ETH_USDT | Bear | 5358 | 13.38 | 2616 | 2742 | 51.62 | 1517 | 28.31 | 49.9 | -116.93 |
| trend_60 | ETH_USDT | Bull | 5259 | 13.13 | 2476 | 2783 | 57.35 | 1557 | 29.61 | 58.83 | 18.52 |
| trend_60 | ETH_USDT | Sideways | 29434 | 73.49 | 14330 | 15104 | 59.14 | 7519 | 25.55 | 63.67 | 53.59 |
| trend_360 | BTC_USDT | Bear | 6172 | 15.35 | 2803 | 3369 | 61.44 | 1452 | 23.53 | 76.24 | 172.99 |
| trend_360 | BTC_USDT | Bull | 5322 | 13.24 | 2294 | 3028 | 61.12 | 1751 | 32.9 | 76.13 | 162.53 |
| trend_360 | BTC_USDT | Sideways | 28715 | 71.41 | 15061 | 13654 | 55.78 | 6667 | 23.22 | 78.09 | 185.9 |
| trend_360 | ETH_USDT | Bear | 6192 | 15.46 | 2907 | 3285 | 64.49 | 1821 | 29.41 | 61.29 | -93.87 |
| trend_360 | ETH_USDT | Bull | 5468 | 13.65 | 2170 | 3298 | 63.02 | 1936 | 35.41 | 64.93 | 98.68 |
| trend_360 | ETH_USDT | Sideways | 28391 | 70.89 | 14345 | 14046 | 55.47 | 6836 | 24.08 | 59.79 | 34.27 |
| trend_1440 | BTC_USDT | Bear | 8865 | 22.05 | 4129 | 4736 | 68.67 | 1779 | 20.07 | 89.38 | 282.92 |
| trend_1440 | BTC_USDT | Bull | 6341 | 15.77 | 3222 | 3119 | 55.04 | 2072 | 32.68 | 81.9 | 167.0 |
| trend_1440 | BTC_USDT | Sideways | 25003 | 62.18 | 12807 | 12196 | 53.93 | 6019 | 24.07 | 72.42 | 153.82 |
| trend_1440 | ETH_USDT | Bear | 8820 | 22.02 | 4024 | 4796 | 66.9 | 2333 | 26.45 | 70.51 | -2.01 |
| trend_1440 | ETH_USDT | Bull | 5372 | 13.41 | 2046 | 3326 | 61.43 | 1853 | 34.49 | 78.63 | 155.4 |
| trend_1440 | ETH_USDT | Sideways | 25859 | 64.57 | 13352 | 12507 | 54.09 | 6407 | 24.78 | 52.41 | -4.51 |
| efficiency_360 | BTC_USDT | Bear | 134 | 0.33 | 120 | 14 | 99.25 | 118 | 88.06 | 99.15 | 381.09 |
| efficiency_360 | BTC_USDT | Bull | 16 | 0.04 | 13 | 3 | 6.25 | 2 | 12.5 | 0.0 | -239.95 |
| efficiency_360 | BTC_USDT | Sideways | 40059 | 99.63 | 20025 | 20034 | 57.24 | 9750 | 24.34 | 77.22 | 177.5 |
| efficiency_360 | ETH_USDT | Bear | 366 | 0.91 | 222 | 144 | 62.02 | 305 | 83.33 | 61.97 | -60.42 |
| efficiency_360 | ETH_USDT | Bull | 19 | 0.05 | 11 | 8 | 31.58 | 0 | 0.0 | undefined | undefined |
| efficiency_360 | ETH_USDT | Sideways | 39666 | 99.04 | 19189 | 20477 | 57.87 | 10288 | 25.94 | 60.95 | 26.52 |
| daily_ma_5_20 | BTC_USDT | Bear | 27997 | 69.63 | 15203 | 12794 | 56.5 | 6859 | 24.5 | 75.71 | 171.3 |
| daily_ma_5_20 | BTC_USDT | Bull | 12212 | 30.37 | 4955 | 7257 | 59.33 | 3011 | 24.66 | 81.47 | 199.33 |
| daily_ma_5_20 | ETH_USDT | Bear | 32255 | 80.53 | 15865 | 16390 | 58.57 | 10063 | 31.2 | 60.67 | 22.52 |
| daily_ma_5_20 | ETH_USDT | Bull | 6274 | 15.67 | 3197 | 3077 | 49.97 | 528 | 8.42 | 66.86 | 51.83 |
| daily_ma_5_20 | ETH_USDT | Sideways | 1522 | 3.8 | 360 | 1162 | 76.35 | 2 | 0.13 | 100.0 | 192.7 |

Trend 360 dòng được dùng làm ví dụ trọng tâm để đọc bảng, không được tuyên bố là window tối ưu. BTC Bull chỉ có 43,10% future labels Up; ETH Bull 39,69%. Điều này cho thấy trend quá khứ và label tương lai là hai đại lượng khác nhau, đồng thời có thể chịu tác động oracle filtering/horizon và mean reversion; chưa thể xác định nguyên nhân.

ETH Bear trend_360 có selective accuracy 61,29% nhưng net **−93,87 bps/mẫu chọn**. Sai lầm có độ lớn return lớn hơn lợi nhuận khi đúng là một khả năng trực tiếp; accuracy không chứa thông tin về độ lớn thắng/thua. Không suy diễn rằng 61% correctness đủ để có edge.

## 4. Confidence và calibration theo regime

Confidence là max(p_up,p_down). Calibration đang xét **top-label correctness**, không phải reliability diagram của riêng probability Up. `calibration_gap` = accuracy toàn nhóm − confidence trung bình, đơn vị điểm phần trăm; ECE dùng 10 bin cố định đều trên [0,1], weighted absolute gap. ECE và confidence min/median/P90/P95/max có đủ cho cả năm định nghĩa trong CSV; bảng dưới minh họa trend_360.

| symbol | regime | confidence_mean | confidence_median | confidence_p90 | calibration_gap | ece_10 | selected_days |
|---|---|---|---|---|---|---|---|
| BTC_USDT | Bear | 0.5774 | 0.5336 | 0.6487 | 3.7 | 3.7 | 13 |
| BTC_USDT | Bull | 0.5941 | 0.5336 | 0.7491 | 1.71 | 3.83 | 14 |
| BTC_USDT | Sideways | 0.5777 | 0.5336 | 0.6735 | -1.99 | 5.2 | 19 |
| ETH_USDT | Bear | 0.5747 | 0.5336 | 0.6487 | 7.02 | 10.21 | 14 |
| ETH_USDT | Bull | 0.5863 | 0.5336 | 0.6487 | 4.39 | 8.52 | 19 |
| ETH_USDT | Sideways | 0.5696 | 0.5336 | 0.6487 | -1.49 | 2.3 | 31 |

ETH Bear có confidence trung bình 0,5747 nhưng accuracy toàn nhóm 64,49% (gap +7,02 điểm phần trăm); ECE 10,21 điểm phần trăm cho thấy sai lệch giữa các confidence bins có thể bù nhau ở mean gap. BTC Sideways gap −1,99 điểm phần trăm. Cùng confidence không thể mặc định có cùng reliability giữa regime; tuy nhiên calibration fit ngay validation nên các con số này không phải chất lượng calibration trên dữ liệu độc lập. `Reliability diagram` chỉ vẽ bin n≥100, bảng CSV giữ mọi bin.

## 5. Up có dễ dự đoán hơn Down không?

Phải phân biệt **precision của dự đoán Up/Down** và **recall theo label thật**. Bảng đầu nhóm theo hướng MLP thực sự dự đoán; accuracy ở đây là directional precision. Bảng sau nhóm theo label thật; accuracy toàn nhóm là class recall. Không đọc selective class recall thành recall toàn bộ cơ hội vì tập mẫu số sau threshold đã đổi.

| symbol | regime | condition | n | accuracy | selected_n | selective_accuracy | net_selected_bps |
|---|---|---|---|---|---|---|---|
| BTC_USDT | Bear | predicted_Down | 4419 | 61.19 | 412 | 93.2 | 160.71 |
| BTC_USDT | Bear | predicted_Up | 1753 | 62.07 | 1040 | 69.52 | 177.85 |
| BTC_USDT | Bull | predicted_Down | 3631 | 63.21 | 730 | 76.03 | 134.14 |
| BTC_USDT | Bull | predicted_Up | 1691 | 56.65 | 1021 | 76.2 | 182.82 |
| BTC_USDT | Sideways | predicted_Down | 21835 | 52.19 | 2595 | 83.28 | 197.69 |
| BTC_USDT | Sideways | predicted_Up | 6880 | 67.18 | 4072 | 74.78 | 178.39 |
| ETH_USDT | Bear | predicted_Down | 3692 | 64.71 | 200 | 54.0 | 4.94 |
| ETH_USDT | Bear | predicted_Up | 2500 | 64.16 | 1621 | 62.18 | -106.06 |
| ETH_USDT | Bull | predicted_Down | 3326 | 69.18 | 324 | 99.38 | 188.18 |
| ETH_USDT | Bull | predicted_Up | 2142 | 53.45 | 1612 | 58.0 | 80.69 |
| ETH_USDT | Sideways | predicted_Down | 19218 | 53.65 | 1052 | 67.4 | 111.24 |
| ETH_USDT | Sideways | predicted_Up | 9173 | 59.28 | 5784 | 58.4 | 20.27 |

Class recall trước threshold:

| symbol | regime | condition | n | accuracy |
|---|---|---|---|---|
| BTC_USDT | Bear | actual_Down | 3369 | 80.26 |
| BTC_USDT | Bear | actual_Up | 2803 | 38.82 |
| BTC_USDT | Bull | actual_Down | 3028 | 75.79 |
| BTC_USDT | Bull | actual_Up | 2294 | 41.76 |
| BTC_USDT | Sideways | actual_Down | 13654 | 83.46 |
| BTC_USDT | Sideways | actual_Up | 15061 | 30.69 |
| ETH_USDT | Bear | actual_Down | 3285 | 72.72 |
| ETH_USDT | Bear | actual_Up | 2907 | 55.18 |
| ETH_USDT | Bull | actual_Down | 3298 | 69.77 |
| ETH_USDT | Bull | actual_Up | 2170 | 52.76 |
| ETH_USDT | Sideways | actual_Down | 14046 | 73.41 |
| ETH_USDT | Sideways | actual_Up | 14345 | 37.91 |

Không có quy luật Up luôn tốt hơn Down. Ví dụ ETH Bull: precision trước threshold Down 69,18%, Up 53,45%; sau threshold Down 99,38% trên 324 selections, Up 58,00% trên 1.612 selections. ETH Bear: Up sau threshold 62,18% trên 1.621 selections nhưng net −106,06 bps; Down 54,00% trên 200 selections, net +4,94 bps. Các số cực đoan có thể do cụm thời gian và biến động lớn, không phải hàng trăm thử nghiệm độc lập. Label Down được recall cao hơn Up trước threshold ở cả ba regime của hai coin; MLP cũng dự đoán Down nhiều hơn, nên không nhầm precision với khả năng nhận diện toàn bộ Up events.

## 6. Thông tin bổ sung sau confidence, symbol và volatility

### Phương pháp thống kê

Hồi quy tuyến tính mô tả với fixed effects của confidence chính xác, làm riêng từng symbol. Dùng hai dummy Bull và Bear, Sideways làm mốc. Trừ trung bình các biến trong từng stratum rồi least squares; hệ số correctness là chênh lệch xác suất đúng điều chỉnh, đơn vị điểm phần trăm. Đây là linear probability association, không phải probability model dùng thực thi và không có diễn giải nhân quả. Không báo p-value IID.

Ba mức kiểm soát: confidence; confidence + quintile volatility_30m theo symbol; thêm ngày UTC. Quintile dùng toàn validation chỉ phục vụ mô tả. Robustness thứ hai dùng log(volatility_30m) và log(volatility_15d) liên tục, kèm confidence hoặc confidence+day. Daily volatility hằng theo symbol/day nên bị hấp thụ bởi day fixed effects, không cố ước lượng một hệ số daily độc lập. Tính log/control bằng float64 tránh rounding tạo ra variation giả trong daily covariate. Các tương quan/hiệu ứng vẫn có thể còn confounding phi tuyến, direction hoặc cấu trúc thời gian chưa mô hình hóa.

### Effect sizes trend_360: adjusted correctness, điểm phần trăm

| symbol | contrast | confidence | confidence_vol | confidence_vol_day | confidence_continuous_vol | confidence_continuous_vol_day |
|---|---|---|---|---|---|---|
| BTC_USDT | Bear minus Sideways | 5.49 | 4.28 | -7.91 | -2.14 | -9.9 |
| BTC_USDT | Bull minus Sideways | 4.55 | 3.41 | 2.92 | 3.1 | 5.07 |
| ETH_USDT | Bear minus Sideways | 8.79 | 8.01 | 5.62 | 2.8 | 4.17 |
| ETH_USDT | Bull minus Sideways | 7.0 | 6.49 | 6.82 | 7.46 | 9.51 |

BTC Bear có +5,49 điểm phần trăm khi chỉ kiểm soát confidence, nhưng −7,91 khi thêm vol-quintile/day; với continuous vol/day là −9,90. ETH Bull giữ dấu dương qua các đặc tả, từ +7,00 đến +9,51 điểm phần trăm; ETH Bear vẫn dương nhưng mức thay đổi đáng kể. Đây là bằng chứng heterogeneity, **chưa phải tín hiệu độc lập ổn định đủ để giao dịch**.

Với continuous vol/day controls, adjusted net trên **toàn bộ cohort, chưa threshold** của Bull so với Sideways chỉ +4,60 bps BTC, +8,06 ETH; Bear −67,62 BTC và −23,90 ETH. Hệ số correctness dương không đảm bảo hiệu ứng return dương. Xem `conditional effects` và `continuous controls`. Không dùng các hồi quy này để thay probability hoặc threshold.

### Tương quan regime score với volatility

| definition | symbol | spearman_score_vol30 | spearman_abs_score_vol30 |
|---|---|---|---|
| trend_60 | BTC_USDT | -0.077 | 0.117 |
| trend_60 | ETH_USDT | -0.104 | 0.078 |
| trend_360 | BTC_USDT | -0.131 | 0.308 |
| trend_360 | ETH_USDT | -0.173 | 0.265 |
| trend_1440 | BTC_USDT | -0.163 | 0.063 |
| trend_1440 | ETH_USDT | -0.18 | 0.191 |
| efficiency_360 | BTC_USDT | -0.129 | 0.311 |
| efficiency_360 | ETH_USDT | -0.173 | 0.278 |
| daily_ma_5_20 | BTC_USDT | -0.09 | -0.077 |
| daily_ma_5_20 | ETH_USDT | -0.296 | 0.324 |

Trend_360 signed score có Spearman với volatility_30m −0,131 BTC, −0,174 ETH; |score| tương quan +0,309 và +0,265. Không phải đồng nhất hoàn toàn nhưng có liên hệ vừa phải với độ mạnh trend. Vì trend đã chuẩn hóa bằng realized quadratic variation, vừa có quan hệ cấu trúc vừa có quan hệ thực nghiệm; không coi việc chia denominator đã loại hết volatility. MA gap và volatility cũng khác nhau theo asset.

## 7. Stability, switching và sensitivity

Switching tính trên **mọi micro row trong khoảng lịch validation mỗi coin**, không chỉ những hàng oracle-filtered; tránh nhầm gap giữa các mẫu được chọn thành regime transition. Run length có hai run biên bị censor. Median run ngắn phản ánh nhiều lần flicker quanh ranh giới ngay cả khi có một số run dài; không đồng nghĩa thị trường thay regime kinh tế mỗi vài phút.

| definition | symbol | switches | switch_rate | median_run_rows | p90_run_rows |
|---|---|---|---|---|---|
| trend_60 | BTC_USDT | 5341 | 7.06 | 4.0 | 43.0 |
| trend_360 | BTC_USDT | 2314 | 3.06 | 4.0 | 76.0 |
| trend_1440 | BTC_USDT | 1147 | 1.52 | 3.0 | 88.3 |
| efficiency_360 | BTC_USDT | 44 | 0.06 | 4.0 | 4911.2 |
| daily_ma_5_20 | BTC_USDT | 2 | 0.0 | 25830.0 | 31446.8 |
| trend_60 | ETH_USDT | 5222 | 7.01 | 4.0 | 45.0 |
| trend_360 | ETH_USDT | 2244 | 3.01 | 3.0 | 89.0 |
| trend_1440 | ETH_USDT | 1000 | 1.34 | 4.0 | 86.0 |
| efficiency_360 | ETH_USDT | 38 | 0.05 | 5.0 | 10075.4 |
| daily_ma_5_20 | ETH_USDT | 4 | 0.01 | 15712.0 | 28311.0 |

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

- `Regime metrics, đầy đủ confidence distribution`, `direction metrics`, `weekly metrics`.
- `Metadata/checksums`: input hashes trước/sau không đổi, không đọc test predictions, không fit threshold.
- Script [analyze_validation.py](../research_regime_analysis/analyze_validation.py) và [robustness.py](../research_regime_analysis/robustness.py), [write_report.py](../research_regime_analysis/write_report.py). Lệnh lần lượt: `reproduction_setup/venv/Scripts/python.exe research_regime_analysis/analyze_validation.py`, rồi tương tự với `robustness.py` và `write_report.py`. Script đầu từ chối overwrite output hiện tại; để tái lập vào thư mục mới cần đổi hằng OUT trước khi chạy, không xóa kết quả cũ.

Runtime phân tích chính: 13.72 giây; chưa tính bước robustness và dựng báo cáo. Không cài thư viện, không train lại MLP, không thay source tác giả, dataset hay kết quả Fixed/A/B cũ.

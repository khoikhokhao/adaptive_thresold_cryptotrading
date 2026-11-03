# Data, features và labels

## 1. Nguồn và cấu trúc raw data

- Macro: [Top 100 Cryptocurrency](https://www.kaggle.com/datasets/imtkaggleteam/top-100-cryptocurrency-2020-2025), daily history2018–2025. Một dòng là symbol×ngày, raw columns: `symbol,date,open,high,low,close,network`. Raw file đã dùng không có measured volume; repo tạo proxy `(high-low)*close*1000`, không gọi đây là traded volume.
- Micro: [Cryptocurrency Order Book Data](https://www.kaggle.com/datasets/ilyazawilsiv/cryptocurrency-order-book-data-asks-and-bids), history2023–2024. Một dòng là order-book snapshot tại phút quan sát, không phải transaction tick hoặc OHLC bar. Columns: `symbol_id,time_exchange_minute`, `asks[k].price,size`, `bids[k].price,size`, k0…49:202 columns.
- Tải thủ công từ hai nguồn chính thức; giữ ZIP/raw gốc bên ngoài Git. Không thay thế bằng synthetic hoặc nguồn khác mà không ghi thành protocol mới.

Đặt input cho processor gốc:

```text
data/
  macro/top_100_cryptos_with_correct_network.csv
  micro/BTC_USDT.csv
  micro/ETH_USDT.csv
```

Để chỉ dùng BTC/ETH, không đặt micro CSV của asset khác trong input directory. Macro có thể giữ nguyên dataset100symbols để bám processor gốc. Symbol micro lấy từ filename; đối chiếu `BINANCE_SPOT_BTC_USDT`/`BINANCE_SPOT_ETH_USDT` và normalized macro symbol.

## 2. Cleaning, features và macro/micro join

Source: [data_preprocessing.py](../data_preprocessing.py).

1. Parse timestamps, normalize symbols, sort và aggregate last observation theo symbol/minute; không tự tạo missing minute rows.
2. Kiểm tra giá/book, crossed market, spread và gaps theo config tác giả. Rolling windows trên những dòng thực có, không xem các phút là liên tục khi dữ liệu có gaps.
3. Macro indicators: daily MA5/10/20, volatility5/15/30, RSI/ATR14 và momentum. Micro: mid-price, spread, imbalance, depth/liquidity, lag và rolling features5/15/30 dòng. Giữ công thức/precision nguồn khi tái lập.
4. Macro date cộng1ngày rồi left-join theo symbol/ngày micro. Indicator có thể có lag riêng trước join; không dùng daily close chưa kết thúc tại thời điểm prediction.
5. Rolling warmup có thể NaN; median imputer và RobustScaler được fit trên train sau split, không fit toàn dataset.

Chạy native entry point ghi `data/processed/unified_dataset.parquet`; đây là input của trainer. Kiểm tra timestamp/symbol uniqueness, ordering, gaps, macro alignment, dtypes và missing values trước train. Không bỏ dữ liệu hoặc đổi precision chỉ để vượt memory guard.

## 3. Return và labels

Source: [TargetEncoder.create_targets](../centralized_training_two_classes.py#L158). Gọi P là midpoint `(best_bid+best_ask)/2`, H là số dòng tiếp theo trong **cùng symbol**:

```math
R_t=\frac{P_{t+H}}{P_t}-1,\qquad R_t^{bps}=10000R_t.
```


```math
Y_t=\begin{cases}
1\;(Up)&R_t^{bps}>\delta\\
0\;(Down)&R_t^{bps}<-\delta\\
2\;(Neutral)&-\delta\le R_t^{bps}\le\delta.
\end{cases}
```


- Paper: H600 phút, δ10bps=0,1%. Implementation dùng H600 **dòng/symbol**, dù flag tên `--horizon_min`; gaps làm elapsed horizon khác600phút.
- Cấu hình BTC/ETH của proposed research: δ100bps=1%, giữ H600 dòng. Đây là thay đổi label đã chọn, không phải exact paper setup.
- Ví dụ toán học, không phải một quan sát thật: P_t100, P_future101,2 cho return120bps→Up khi δ100; P_future100,8 cho80bps→Neutral. R=±100bps vẫn Neutral vì source dùng strict inequalities.
- Những dòng không có future endpoint bị loại. Source dùng buffer future price float32; khi tái lập cần giữ dtype và source calculation, không thay bằng formula implementation khác để làm tròn label.

## 4. Split, feature selection và training input

[TemporalSplitter](../centralized_training_two_classes.py#L602) chia70/15/15 chronological **riêng từng symbol**. Labels được tạo trước split. Với `--two_class_mode`, source giữ Up/Down và loại Neutral trong cả train/validation/test; model chỉ dự đoán hai lớp. Neutral không đồng nghĩa Abstain của threshold.

Train-only MI chọn128features, dùng tối đa100.000 train observations để tính scores, seed42. Sau đó fit imputer/scaler bằng train, apply nguyên artifacts lên validation/test. Feature vector MLP là một row128chiều, không phải một sequence600phút. MLP two-class trong repo dùng sklearn/SGD; khác PyTorch/Adam mô tả trong paper.

Probability p_up là score Up của model/calibrator trên protocol directional-sample này; confidence=max(p_up,p_down). Giữ direction dự đoán rồi quyết định Long/Short/Abstain bằng confidence threshold. Threshold được chọn bằng validation, sau đó đóng băng khi test; không chọn lại theo test performance.

## 5. Evaluation scope

Lọc Neutral bằng future returns tạo oracle-filtered evaluation; future horizon overlap và main legacy split chưa purge endpoints. Đây là giới hạn phải giữ rõ khi so methods, không diễn giải thành capital-constrained portfolio backtest. Không dùng test labels để định nghĩa factor hoặc chọn threshold. Xem [cách chạy](RUNNING.md) và [paper context](../proposoed_method_v2/README.md).

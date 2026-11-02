# Điều kiện tái lập

Repository chứa source paper, proposal v2, analysis và summary artifacts nhỏ. Raw/processed data, matrices, weights và row-level predictions không bundle. Đợt dọn dự án không chạy lại preprocessing, training hoặc benchmark.

## Dữ liệu và baseline

- Macro: [Top 100 Cryptocurrency](https://www.kaggle.com/datasets/imtkaggleteam/top-100-cryptocurrency-2020-2025). Raw gồm symbol/date/OHLC/network; không measured volume, repo tạo proxy volume.
- Micro: [Cryptocurrency Order Book Data](https://www.kaggle.com/datasets/ilyazawilsiv/cryptocurrency-order-book-data-asks-and-bids). Dùng full history BTC_USDT/ETH_USDT, timestamp và 50 bid/ask price-size levels.
- Raw đặt trong `data/macro/`, `data/micro/`. Không dùng synthetic hoặc dataset thay thế.
- Unified: `data/join_full_btc_eth_20261102/unified_dataset.parquet`, 1.008.203×296; SHA256 `9b021ece5a7ae37b7c988c1d0a83e00329527127fe5ebda661d9986845238827`.
- Baseline: horizon600 dòng, deadband100bps, lọc Neutral trên cả ba split; train292.338/val80.260/test65.182; MI top128, train imputer/scaler, sklearn MLP256–128–64 và isotonic validation. [Training history](BTC_ETH_BASELINE_1PCT.md).
- Fixed τ0,64 từ `data/btc_eth_1pct/coverage_10pct/selection.json`; pooled validation coverage≥10%, không chọn bằng test.

## Environment

Dependencies của source tác giả giữ ở `requirements.txt`; research additions ở `requirements-research.txt`. Snapshot local: `reproduction_setup/requirements.freeze.txt`. Đã dùng Windows/Python3.13.3; các guard gọi Windows APIs, cap9GiB và Available RAM≥2GiB. Không có GPU requirement cho sklearn branch.

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
```

## Inputs cần trước khi chạy threshold

| Method | Artifacts local bắt buộc |
|---|---|
| Volatility-B | `research_volatility_threshold/full_validation_test_20261103/`: validation/test decisions, normalization/lock config, metrics |
| Regime-B | `research_regime_threshold/method_a_20261103/`: z, decisions, configs; model/predictions/checksum dependencies |
| ACF-A | Unified, predictions, cached ACF/folds trong `research_autocorrelation_analysis/validation_20261103/`, selection, reference metrics Regime/Volatility |
| Shift-A | `X_train/X_validation/X_test.npy`, metadata/manifests ở `data/btc_eth_1pct/prepared/`, predictions, train PCA reference/folds ở `research_distribution_shift/validation_20261103/`, ACF summaries |

Không đủ inputs thì dừng, không fill bằng dữ liệu giả. Một số helper/reference từ method A cũ còn cần cho methods được chọn; giữ chúng không có nghĩa thêm proposed method.

## Thứ tự tái dựng từ dữ liệu

1. Processor gốc hoặc memory utilities trong `reproduction_setup/` tạo unified BTC/ETH. Có nhiều entry points theo checkpoint; chưa đóng gói một lệnh end-to-end tự động từ raw.
2. Chuẩn bị matrices/model/predictions. `training_memory.py`, `train_btc_eth_1pct.py` là historical wrappers có checks/refuse-overwrite. CLI gốc nằm trong `UPSTREAM_README.md`; example deadband10bps khác run100bps hiện tại.
3. Volatility: `reproduction_setup/volatility_validation_analysis.py`, các helpers trong `research_volatility_threshold/` tạo A/reference caches trước B. Regime: `research_regime_analysis/analyze_validation.py`, robustness, sau đó Regime-A caches trước B.
4. ACF: `research_autocorrelation_analysis/explore.py`, `conditional_analysis.py`, `summarize.py`. Shift: `research_distribution_shift/build_scores.py`, `describe.py`, `conditional_analysis.py`, `robustness.py`. Kiểm tra input constants và caches trước khi chạy.
5. Threshold scripts theo dependency order: Volatility-B → Regime-B → ACF-A → Shift-A. Lệnh ghi output mới có trong [proposal](../proposoed_method_v2/README.md). Không dùng test để chọn winner.

Fresh clone **chưa chạy benchmark ngay** nếu chưa tái tạo hoặc cung cấp cached data/model/predictions. Không claim đã xác minh clean-room rerun; đợt này chỉ kiểm tra cấu trúc, links, syntax, dependency paths và preservation của upstream source.

## Artifacts và giới hạn

`metrics_validation_test.csv`, `REPORT.md` và locked configs trong `research_*threshold/` là kết quả thực tế lịch sử; các CSV/figures analysis là số đo validation. Giữ riêng khỏi bảng kịch bản của v2. Parquet/joblib/npy, environment và runtime logs bị ignore nhưng có thể còn local.

Không có position/capital engine; không cộng overlapping sample returns thành portfolio PnL/annualized Sharpe. Oracle filtering, horizon theo dòng, missing purge main split, validation reuse và test đã xem vẫn tồn tại; không confirmatory OOS.

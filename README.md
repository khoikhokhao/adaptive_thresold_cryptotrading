# Adaptive Confidence Thresholds for Cryptocurrency Prediction

Phần mở rộng nghiên cứu của *Machine Learning Analytics for Blockchain-Based Financial Markets: A Confidence-Threshold Framework for Cryptocurrency Price Direction Prediction*, Kuznetsov et al., Applied Sciences 2025, DOI [10.3390/app152011145](https://doi.org/10.3390/app152011145).

Repository kết hợp **source gốc của tác giả** với bốn hướng confidence-threshold selection của Phạm Minh Khôi trên cùng MLP predictions BTC/ETH. Không coi implementation hiện tại là exact reproduction của paper.

## Đọc từ đâu?

| Thành phần | Tài liệu |
|---|---|
| Paper, phương pháp gốc và research gap | [PDF/context](proposoed_method_v2/README.md) |
| Volatility — Method B | [Proposal và lệnh chạy](proposoed_method_v2/volatility_B.md) |
| Market Regime — Method B | [Proposal và lệnh chạy](proposoed_method_v2/market_regime_B.md) |
| Autocorrelation — Method A | [Proposal và lệnh chạy](proposoed_method_v2/autocorrelation_A.md) |
| Distribution Shift — Method A | [Proposal và lệnh chạy](proposoed_method_v2/distribution_shift_A.md) |
| Evidence validation và số liệu phân tích thực | [Analysis](analysis/README.md) |
| Environment, inputs và dependency order | [Reproducibility](docs/REPRODUCIBILITY.md) |
| Hướng dẫn nguyên bản của tác giả | [Upstream README](docs/UPSTREAM_README.md) |

## Cấu trúc và attribution

- `data_preprocessing.py`, `centralized_training_two_classes.py`, `analyze_centralized_model.py`: đường centralized của tác giả, giữ nguyên source. Những source khác của repo gốc, gồm sweep và federated/privacy, cũng được giữ; chúng không phải proposed methods mới.
- `proposoed_method_v2/`: PDF và bốn proposal đã chọn. Bảng proposed v2 là kịch bản minh họa theo note cuối từng tài liệu; không thay thế số liệu thực trong analysis/benchmark summaries.
- `analysis/`: bốn exploratory reports. Các thư mục `research_*analysis/` và `research_distribution_shift/` giữ source phân tích, bảng tổng hợp và figures.
- `research_*threshold/`: code selection và summary artifacts thực tế. `method_a.py`, `run_regime_a.py` còn làm helper/đối chứng cho những methods được chọn.
- `reproduction_setup/`, `baseline_reproduction/`: memory utilities, train wrapper và guard, tách khỏi source tác giả.
- `data/`, matrices, models, prediction Parquets, virtual environment và logs là artifacts local, không đưa lên GitHub.

## Environment và cách chạy

Môi trường BTC/ETH đã dùng Python 3.13.3 trên Windows. Tạo môi trường riêng nếu chưa có:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
```

Lệnh trong proposal dùng executable local cũ `reproduction_setup/venv/Scripts/python.exe`; nếu tạo `.venv`, thay executable bằng `.venv/Scripts/python.exe`. Một số guards dùng Windows memory APIs; chưa xác minh chạy nguyên bản trên Linux/macOS.

1. Chuẩn bị đúng raw macro/micro và unified BTC/ETH theo [hướng dẫn](docs/REPRODUCIBILITY.md).
2. Chuẩn bị model, predictions và cached matrices đúng baseline: deadband **100bps**, horizon **600 dòng/symbol**, 128 inputs, sklearn/SGD và isotonic validation. Paper dùng deadband10bps và mô tả PyTorch/Adam.
3. Tái tạo analysis/caches và summary dependencies theo thứ tự trong hướng dẫn. Fresh clone không bao gồm dataset/model để bỏ qua bước này.
4. Dùng lệnh từng proposal: chọn threshold trên validation, khóa rồi đánh giá test; output vào thư mục mới. Threshold scripts không train lại MLP và không chọn parameters bằng test.

Fixed τ=0,64 được chọn trên validation, pooled minimum coverage10%, cost1bps. Cohort còn oracle filtering, labels overlap, missing purge và validation reuse; test đã xem nên kết quả thực hiện tại chỉ retrospective/exploratory, không confirmatory OOS hoặc portfolio backtest.

## Nguồn gốc và giấy phép

Upstream: [KuznetsovKarazin/crypto-confidence-execution](https://github.com/KuznetsovKarazin/crypto-confidence-execution), commit `21904283bb7631e1d9e9ec0d254ce318ae0861ab`. Source gốc được đối chiếu với Git commit, chỉ bỏ qua CRLF/LF. Giữ [LICENSE](LICENSE) của upstream. PDF công bố CC BY4.0, giữ DOI/attribution. Chưa xác minh novelty của proposed methods bằng literature review.

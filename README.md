# Adaptive Thresholds for Cryptocurrency Direction Prediction

Source centralized của paper *Machine Learning Analytics for Blockchain-Based Financial Markets: A Confidence-Threshold Framework for Cryptocurrency Price Direction Prediction* (Applied Sciences, 2025), cùng bốn proposed methods: Volatility-B, Regime-B, ACF-A và Shift-A.

## Nội dung repository

| Nội dung | Tài liệu |
|---|---|
| Paper gốc và context | [PDF/context](proposoed_method_v2/README.md) |
| Volatility-B | [Report](proposoed_method_v2/volatility_B.md) |
| Regime-B | [Report](proposoed_method_v2/market_regime_B.md) |
| ACF-A | [Report](proposoed_method_v2/autocorrelation_A.md) |
| Shift-A | [Report](proposoed_method_v2/distribution_shift_A.md) |
| Exploratory analysis | [Bốn analysis reports](analysis/README.md) |
| Data, schema, features và labels | [Data & Labels](docs/DATA_AND_LABELS.md) |
| Environment, preprocessing, training và chạy từng method | [Running](docs/RUNNING.md) |

Mỗi method report chỉ giữ ba link: context, analysis và source. Hướng dẫn chạy nằm riêng trong `docs/`, không nằm trong report.

## Source

- `data_preprocessing.py`, `centralized_training_two_classes.py`, `analyze_centralized_model.py`: source centralized của tác giả, giữ nguyên. `sweep_two_class.py`, `sweep_results_aggregator.py`, `plot_tau_curves.py` giữ để dùng CLI/sweep gốc.
- `research_*analysis/`, `research_distribution_shift/`: source phân tích.
- `research_*threshold/`: source bốn methods và các helper cần thiết; helper A còn được import bởi B không phải một proposed method bổ sung.
- `reproduction_setup/`, `baseline_reproduction/`: source utilities dữ liệu/bộ nhớ, train wrapper và guard.
- `analysis/`: chỉ báo cáo Markdown. `proposoed_method_v2/`: PDF/context và bốn report v2.

Không đưa dataset, model, test/validation predictions, output CSV/JSON/plots benchmark, báo cáo benchmark cũ, test scripts hoặc nhánh federated/privacy lên bản hiện tại. Analysis có 6 hình validation được chọn; không phục hồi các output benchmark. Các artifacts phải được tạo hoặc cung cấp local trước khi chạy; fresh clone chưa có cached model/predictions.

## Bắt đầu

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
```

Đọc Data & Labels rồi Running để chuẩn bị inputs và chạy từng stage. Môi trường đã dùng Windows/Python3.13.3; guard dùng Windows APIs. Bản repo hiện tại không tự chạy training hoặc benchmark khi đọc tài liệu.

Upstream: [KuznetsovKarazin/crypto-confidence-execution](https://github.com/KuznetsovKarazin/crypto-confidence-execution), commit `21904283bb7631e1d9e9ec0d254ce318ae0861ab`. Paper DOI [10.3390/app152011145](https://doi.org/10.3390/app152011145); PDF CC BY4.0. Giữ [LICENSE](LICENSE). Các bảng proposed v2 là mục tiêu/kịch bản theo context, không phải kết quả thực nghiệm. Chưa xác minh novelty hoặc confirmatory OOS.

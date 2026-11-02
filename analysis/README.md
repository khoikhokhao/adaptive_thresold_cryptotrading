# Exploratory analysis — validation BTC/ETH

Các báo cáo dưới đây ghi kết quả phân tích đã lưu trên validation; không dùng bảng metrics minh họa của proposal làm evidence.

| Hướng | Report | Source |
|---|---|---|
| Volatility | [Validation analysis](VOLATILITY_VALIDATION_ANALYSIS.md) | [analysis.py](../reproduction_setup/volatility_validation_analysis.py) |
| Market Regime | [Regime analysis](REGIME_EXPLORATORY_ANALYSIS.md) | [analyze_validation.py](../research_regime_analysis/analyze_validation.py) |
| Autocorrelation | [ACF analysis](AUTOCORRELATION_EXPLORATORY_ANALYSIS.md) | [explore.py](../research_autocorrelation_analysis/explore.py) |
| Distribution Shift | [Shift analysis](DISTRIBUTION_SHIFT_EXPLORATORY_ANALYSIS.md) | [build_scores.py](../research_distribution_shift/build_scores.py) |

CSV/JSON/figures nhỏ được giữ để kiểm chứng số liệu; matrices, model/reference joblib và row-level predictions chuẩn bị local. Volatility evidence được sao chép vào `analysis/volatility/`; source vẫn ghi output local vào `data/volatility_validation_btc_eth/`.

Correlation hoặc relative logloss gain không tự chứng minh trading utility/concept drift. Xem [context](../proposoed_method_v2/README.md) và [điều kiện tái lập](../docs/REPRODUCIBILITY.md).

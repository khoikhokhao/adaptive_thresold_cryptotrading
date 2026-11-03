# Exploratory analysis — validation BTC/ETH

Chỉ giữ báo cáo phân tích, không đính kèm output files hoặc benchmark lịch sử.

| Hướng | Analysis report | Source |
|---|---|---|
| Volatility | [Report](VOLATILITY_VALIDATION_ANALYSIS.md) | [Source](../reproduction_setup/volatility_validation_analysis.py) |
| Market Regime | [Report](REGIME_EXPLORATORY_ANALYSIS.md) | [Source](../research_regime_analysis/analyze_validation.py) |
| Autocorrelation | [Report](AUTOCORRELATION_EXPLORATORY_ANALYSIS.md) | [Source](../research_autocorrelation_analysis/explore.py) |
| Distribution Shift | [Report](DISTRIBUTION_SHIFT_EXPLORATORY_ANALYSIS.md) | [Source](../research_distribution_shift/build_scores.py) |

Các bảng nằm trong report là evidence validation đã lưu; không dùng bảng kịch bản trong proposal làm evidence. Những file CSV/cache được nhắc trong nội dung là artifacts local khi chạy analysis, không bundle vào repository. Correlation/logloss gain không tự chứng minh economic edge. Xem [context](../proposoed_method_v2/README.md) và [cách chạy](../docs/RUNNING.md).

# Exploratory analysis — validation BTC/ETH

Giữ báo cáo phân tích và 6 hình validation được chọn trong `figures/`; không đính kèm output CSV/JSON, predictions hoặc benchmark lịch sử.

| Hướng | Analysis report | Source |
|---|---|---|
| Volatility | [Report](VOLATILITY_VALIDATION_ANALYSIS.md) | [Source](../reproduction_setup/volatility_validation_analysis.py) |
| Market Regime | [Report](REGIME_EXPLORATORY_ANALYSIS.md) | [Source](../research_regime_analysis/analyze_validation.py) |
| Autocorrelation | [Report](AUTOCORRELATION_EXPLORATORY_ANALYSIS.md) | [Source](../research_autocorrelation_analysis/explore.py) |
| Distribution Shift | [Report](DISTRIBUTION_SHIFT_EXPLORATORY_ANALYSIS.md) | [Source](../research_distribution_shift/build_scores.py) |

Các bảng analysis trình bày số liệu validation; đề xuất nghiên cứu nằm riêng trong report v2. Những file CSV/cache được nhắc trong nội dung là artifacts local khi chạy analysis, không bundle vào repository. Correlation/logloss gain không tự chứng minh economic edge. Xem [context](../proposoed_method_v2/README.md) và [cách chạy](../docs/RUNNING.md).

## Các hình được bổ sung

- Volatility: accuracy/coverage/net theo volatility quantiles.
- Regime: performance theo Bull/Bear/Sideways và khác biệt predicted Up/Down, tách BTC/ETH.
- ACF: performance theo absolute-ACF quintiles và độ ổn định theo tuần.
- Distribution Shift: PCA-residual incremental gain qua folds, đi kèm kiểm tra comparator health.

Ba hình phục hồi nguyên bản từ analysis đã lưu; ba hình tập trung dựng từ các CSV summary validation đã lưu. Không đọc test predictions, không fit thêm model hoặc chạy lại benchmark. Các số trên hình thuộc exploratory analysis thật; proposal v2 không thay đổi.

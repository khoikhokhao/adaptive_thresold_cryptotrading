# Method A so với fixed threshold 0,64 — coverage pooled

Đã sửa cách diễn giải constraint theo yêu cầu: coverage tối thiểu 10% chỉ trên tổng mẫu của đoạn validation chọn tham số; không bắt từng coin đạt 10%. Không có coverage constraint trên đoạn kiểm tra. Giữ nguyên objective tối đa mean net/mẫu chọn, grid 11 beta, predictions, labels, phí 1 bps và fixed threshold cũ. Lần chạy trước được giữ nguyên để truy vết.

Kết quả: beta tối ưu bằng 0. Method A trở về threshold 0,64 cho mọi mẫu. Đã xác minh boolean selection trùng tuyệt đối với baseline trên toàn bộ 80.260 mẫu validation, không chỉ metrics giống nhau. Không chọn beta khác zero chỉ để tạo ra một adaptive method khác baseline.

## So sánh cùng cohort validation cũ

Cả hai chọn 20.463 / 80.260 mẫu: coverage 25,4959%; selective accuracy 68,9342%; net/mẫu chọn khoảng 99,1812 bps. Bảng riêng BTC/ETH nằm trong full_validation_comparison.csv. Đây là so sánh mô tả toàn validation, không phải đoạn độc lập dùng chọn lại beta.

## Chọn tham số và kiểm tra theo thời gian

Trên V_select 19.061 mẫu, beta=0 đạt mean net cao nhất trong grid: 183,9287 bps/mẫu chọn; coverage 18,1627%; selective accuracy 81,0225%. Mọi beta đều vượt minimum coverage pooled 10%. Đây là đoạn dùng chọn beta.

Sau khi khóa beta và các cutoff coverage-matched, đánh giá V_check 23.708 mẫu: cả hai chọn 14.371 mẫu, coverage 60,6167%; selective accuracy 62,8488%; net/mẫu chọn 63,7105 bps; expected net/mẫu đủ điều kiện 38,6192 bps. V_check vẫn là validation, không phải test. Bảng native_metrics.csv có chi tiết BTC/ETH. Không dùng V_check để đổi hệ số.

Coverage-matched dùng cùng beta=0 nên score A và fixed giống nhau, các kết quả từng operating point cũng giống nhau; coverage thực tế vẫn có thể drift giữa V_select và V_check. Không ép coverage bằng quantile trên đoạn kiểm tra.

## Kết luận

Method A chưa cải thiện fixed 0,64 theo objective đã chọn; phương án được chọn chính là không điều chỉnh theo volatility. Giữ fixed baseline. Việc beta âm có expected net cao hơn trên V_select là một objective khác, không dùng nó để đổi cách chọn hậu nghiệm.

Cohort đã oracle-filtered theo |future return|>1%; return các mẫu chồng lấn. Model/calibrator và fixed threshold đã dùng validation; kết quả là retrospective/exploratory, không phải trading backtest hoặc confirmatory OOS. Không đọc predictions test và không train lại MLP.

Lệnh: `reproduction_setup/venv/Scripts/python.exe research_volatility_threshold/method_a.py --coverage-scope pooled --out research_volatility_threshold/method_a_pooled_20261103`

Artifacts: locked_config.json, setup_manifest.json, beta_grid_select.csv, split_counts.csv, native_metrics.csv, full_validation_comparison.csv, coverage_matched_metrics.csv, weekly_metrics.csv, validation_decisions.parquet, selection_grid.png, locked_coverage_curves.png, status.json và resources.jsonl.

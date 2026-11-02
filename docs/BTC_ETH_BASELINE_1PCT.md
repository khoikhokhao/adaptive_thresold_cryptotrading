# BTC/ETH — baseline repo, deadband 1%

Ngày 02/11/2026. Run hoàn tất trên unified dataset BTC/ETH hiện có; source tác giả giữ nguyên. Phạm vi là tái hiện code repo với deadband do người dùng chọn, không phải exact reproduction của paper.

## Cấu hình

- Horizon 600 hàng/symbol; deadband 100 bps. Lọc Neutral trên cả train/validation/test theo source.
- Train 292.338, validation 80.260, test 65.182 mẫu; features 128 từ mutual information trên 100.000 train samples, seed 42. Imputer median và RobustScaler fit trên train.
- MLP sklearn/SGD của tác giả, hidden 256–128–64, LR 0,001, sample weights balanced, batch thực tế 200. Budget 20 outer epochs, patience 10.
- Isotonic calibration trên validation; threshold được tối ưu **trên validation** bằng mean net profit trên mẫu được chọn, chi phí 1 bps, không đặt minimum coverage. Grid 0,50–0,94 bước 0,02, gồm 23 điểm. Không tối ưu threshold trên test.

`training_memory.py` và `train_btc_eth_1pct.py` trong `reproduction_setup/` là wrapper riêng để xử lý bộ nhớ và ghi artifacts, gọi lại logic model của tác giả. Không sửa `centralized_training_two_classes.py`. Dataset checksum và matrix hashes được kiểm tra trước train; validation transform được đối chiếu chính xác với matrix đã lưu.

## Kết quả

Early stopping tại outer epoch 11; best checkpoint là epoch 1. Tổng 12 lời gọi fit, 78 actual passes. Training mất 368,9 giây, peak process-tree working set 0,995 GiB; preparation mất 165,93 giây, peak 1,846 GiB. Memory guard 9 GiB không kích hoạt.

| Split | Samples | Threshold | Executions | Coverage | Selective accuracy | Mean net profit |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 80.260 | 0,94 | 1 | 0,001246% | 100% trên một mẫu | 199,7403 bps trên một mẫu |
| Test | 65.182 | 0,94 | 0 | 0% | Undefined | Undefined |

Accuracy trên toàn bộ các test predictions đã lọc Neutral là 51,0340%; confidence test cao nhất 0,938095. Một execution trên validation không đủ để đánh giá độ ổn định threshold. Native JSON ghi 0 accuracy và −1 bps khi không có executions: đó là placeholders của code, không phải accuracy/profit thực tế trên một tập có giao dịch. Native Sharpe trên validation với một mẫu cũng không có ý nghĩa.

Threshold 0,94 là điểm có mean profit cao nhất theo grid validation và objective không ràng buộc coverage của tác giả. Không đổi threshold sau khi xem test. Calibration và threshold selection cùng dùng validation theo code gốc.

Cả ba tập đã loại Neutral bằng future returns; đây là **oracle-filtered evaluation**, không phải backtest có thể giao dịch thực tế. Các prediction horizons chồng lấn; không coi tổng sample returns là PnL danh mục. Test lịch sử đã được xem trong nghiên cứu trước nên không phải confirmatory holdout.

## Artifacts và xác minh

- Dữ liệu, matrices, MI scores và scaler: `data/btc_eth_1pct/prepared/`.
- Best model, calibrated classifier, 12 checkpoints, config, history, validation/test predictions và metrics: `data/btc_eth_1pct/training/`.
- Grid validation: `data/btc_eth_1pct/training/tau_optimization_results.csv`; threshold: `decision_threshold.json`.
- Guard logs và memory samples: `baseline_reproduction/evidence/btc_eth_1pct_prepare_*` và `btc_eth_1pct_train_*`.
- Kiểm tra 23 grid points, argmax validation, số rows, probabilities hữu hạn/tổng bằng 1, execution masks và hash source tác giả: PASS.

Chưa thay threshold dựa trên test, chưa train lại và chưa triển khai adaptive method.

## Baseline threshold tham chiếu

Sau run đầu tiên, threshold được chọn lại chỉ trên validation với minimum coverage pooled 10%, grid 0,50–0,94 bước 0,02; nghiệm τ=0,64. Đây là fixed baseline của bốn proposed methods. Actual metrics được giữ trong các thư mục benchmark được liệt kê ở docs/REPRODUCIBILITY.md.

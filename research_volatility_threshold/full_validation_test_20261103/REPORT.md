# Method A và fixed τ=0,64 trên cùng validation/test gốc

## Protocol được sửa theo yêu cầu

Những lần trước tự chia validation để chọn beta không cùng phạm vi baseline cũ. Lần này chọn beta trên **toàn bộ 80.260 validation samples gốc**, coverage pooled ≥10%, objective maximum mean net bps/mẫu chọn, grid −0,10 đến +0,10 bước 0,02. Không ràng buộc coverage từng coin hoặc trên test. Không train lại MLP, đổi predictions/labels/phí hoặc baseline. Giữ nguyên kết quả các lần trước để truy vết; kết luận beta=0 ở lần chia nhỏ không phải kết quả của protocol này.

Tái sử dụng median/IQR volatility riêng từng coin đã fit trên phần validation đầu ở lần trước, giữ nguyên các thống kê đó trước khi chọn beta; không chuẩn hóa trên test. Threshold = clip(0,64 + beta × tanh((volatility_30m − median)/IQR), 0,50, 0,95). Dùng đúng giá trị floating point fixed trong selection.json: 0.6400000000000001. Đây là các cohort gốc không thêm purge trong lần so sánh này để giữ cùng phạm vi baseline cũ.

**Beta chọn trên full validation: −0,06.** Lưu `locked_config_before_test.json` trước lần đọc predictions test đầu tiên trong script. Sau đó áp dụng đúng beta đó lên **65.182 test samples gốc**, không tối ưu lại beta hoặc threshold theo test. Lệnh: `reproduction_setup/venv/Scripts/python.exe research_volatility_threshold/full_validation_test.py`. Script từ chối ghi đè thư mục kết quả.

## Tổng hợp

| Tập | Phương pháp | Tổng mẫu | Mẫu chọn | Coverage | Selective accuracy | Net/mẫu chọn (bps) | Expected net/mẫu đủ điều kiện (bps) |
|---|---|---:|---:|---:|---:|---:|---:|
| Validation | Fixed 0,64 | 80.260 | 20.463 | 25,50% | 68,93% | 99,18 | 25,29 |
| Validation | A, beta=−0,06 | 80.260 | 18.899 | 23,55% | 71,93% | 117,12 | 27,58 |
| Test | Fixed 0,64 | 65.182 | 7.984 | 12,25% | 68,75% | 87,13 | 10,67 |
| Test | A, beta=−0,06 | 65.182 | 7.434 | 11,41% | 68,39% | 89,52 | 10,21 |

## Test theo symbol

| Symbol | Method | Tổng mẫu | Mẫu chọn | Coverage | Selective accuracy | Net/mẫu chọn (bps) | Expected net/mẫu đủ điều kiện (bps) |
|---|---|---:|---:|---:|---:|---:|---:|
| BTC | Fixed | 29.004 | 3.808 | 13,13% | 49,13% | −44,29 | −5,82 |
| BTC | A | 29.004 | 3.509 | 12,10% | 46,17% | −55,08 | −6,66 |
| ETH | Fixed | 36.178 | 4.176 | 11,54% | 86,64% | 206,97 | 23,89 |
| ETH | A | 36.178 | 3.925 | 10,85% | 88,25% | 218,79 | 23,74 |

## Diễn giải

Trên validation, A tốt hơn theo cả selective accuracy, net/mẫu chọn và expected net/mẫu đủ điều kiện, với coverage thấp hơn. Sang test, A tăng net/mẫu chọn khoảng 2,39 bps, nhưng giảm selective accuracy khoảng 0,36 điểm phần trăm và expected net/mẫu đủ điều kiện khoảng 0,46 bps. BTC tệ hơn; ETH accuracy và net/mẫu chọn tốt hơn nhưng expected net/mẫu đủ điều kiện giảm nhẹ vì chọn ít mẫu hơn.

**Chưa có cơ sở nói A thắng fixed tổng thể.** Hai operating points không matched coverage; không gán mọi khác biệt cho chất lượng ranking. Không đổi beta hoặc objective sau khi thấy bảng test này. Bảng full_validation_beta_grid.csv gồm tất cả beta chỉ trên validation; không quét beta khác trên test.

## Giới hạn

Oracle-filtered evaluation: đã loại Neutral theo |future return|>1%. Horizon 600 dòng không đảm bảo 600 phút; overlap labels, thiếu purge và sự tái sử dụng validation của baseline vẫn tồn tại. Net là signed sample return trừ 1 bps, không phải portfolio PnL. Test này đã được xem từ nghiên cứu trước nên kết quả là retrospective/exploratory; việc khóa trước test trong lần chạy này không biến nó thành holdout độc lập. Không tuyên bố significance hoặc confirmatory OOS.

Artifacts: `full_validation_beta_grid.csv`, `locked_config_before_test.json`, `metrics_validation_test.csv`, `validation_decisions.parquet`, `test_decisions.parquet`, `completion.json`. File completion lưu runtime, memory, hash cấu hình đã khóa và test predictions; source tác giả và dữ liệu/model cũ giữ nguyên.

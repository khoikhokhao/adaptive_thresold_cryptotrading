# Hướng dẫn chạy

Tài liệu này tập trung lệnh chạy; report v2 chỉ trình bày nghiên cứu. Không chạy bất kỳ stage nào trong đợt chỉnh repository. Datasets, models, predictions và generated artifacts không bundle; script cần chúng thì phải tạo/cung cấp local trước.

## Environment

Đã dùng Python3.13.3 trên Windows. `requirements.txt` giữ dependencies upstream; `requirements-research.txt` thêm dependencies research. Snapshot environment ở `reproduction_setup/requirements.freeze.txt`.

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
```

Trong các lệnh method lịch sử bên dưới, nếu không dùng môi trường cũ `reproduction_setup/venv`, thay executable bằng `.venv/Scripts/python.exe` và thay `Set-Location` bằng thư mục clone của bạn. Guards dùng Windows APIs, cap9GiB và minimum Available RAM2GiB.

## Data → Preprocessing

Tải và tổ chức raw theo [Data & Labels](DATA_AND_LABELS.md). Entry point preprocessing gốc nhận paths trong block cuối `data_preprocessing.py`, không có CLI `--macro`/`--micro`:

```powershell
& .\.venv\Scripts\python.exe .\data_preprocessing.py
```

Mặc định đọc `data/macro/top_100_cryptos_with_correct_network.csv`, tất cả CSV trong `data/micro/`, ghi `data/processed/`. Để universe BTC/ETH, chỉ đặt hai micro CSV BTC/ETH ở input directory; giữ bản raw bên ngoài. Native pipeline có thể cần nhiều RAM. Memory utilities ở `reproduction_setup/` phục vụ từng checkpoint; chưa đóng gói một lệnh raw→full dataset cho mọi máy. Không hứa cap9GiB cho native preprocessing chỉ từ command trên.

## Labels → Centralized MLP

Labels được trainer tạo từ unified; không cần tạo CSV labels thủ công. Ví dụ cấu hình BTC/ETH100bps, horizon600 dòng, optimize threshold trên validation với coverage≥10%:

```powershell
& .\.venv\Scripts\python.exe .\centralized_training_two_classes.py `
  --unified data/processed/unified_dataset.parquet `
  --out research_runs/centralized_btc_eth `
  --task classification --model mlp `
  --horizon_min 600 --deadband_bps 100 --two_class_mode `
  --use_robust_scaler --calibrate_probabilities `
  --top_k_features 128 --batch_size 1024 --epochs 20 --seed 42 `
  --profit_cost_bps 1.0 --confidence_grid 0.50:0.94:0.02 `
  --optimize_threshold_for_profit --optimize_tau_by profit_with_min_coverage `
  --min_coverage 0.10
```

Paper dùng deadband10bps; thay100→10 nếu tái hiện cấu hình label của paper, không nhầm hai protocol. Sklearn branch có actual batch khác CLI batch; xem context. CLI gốc hỗ trợ threshold optimization nhưng không trực tiếp tạo toàn bộ cached Parquets/matrices của research wrappers. Lệnh này không cam kết sinh đúng artifacts/hashes của historical baseline hoặc tự khôi phục mọi cache cho methods.

## Chuẩn bị inputs cho analysis và methods

- Baseline được chọn dùng các paths local `data/btc_eth_1pct/prepared/`, `data/btc_eth_1pct/training/`, `data/btc_eth_1pct/coverage_10pct/selection.json`; các wrappers `training_memory.py`, `train_btc_eth_1pct.py` có schema/count/hash checks và refuse-overwrite.
- Analysis source order: volatility analysis → Regime analysis/robustness → ACF explore/conditional/summarize → Shift build_scores/describe/conditional/robustness. Mỗi source khai báo input constants và output paths, phải kiểm tra trước khi chạy; scripts viết đè output cũ ở một số stages.
- Một số threshold scripts import helper A và đọc reference outputs từ stage trước. Cần tạo các artifacts local theo source trước khi chạy Volatility-B → Regime-B → ACF-A → Shift-A. Repository không chứa các outputs đó; fresh clone không chạy methods ngay mà bỏ qua bước chuẩn bị.
- Không dùng test để chọn factors, coefficients hoặc cutoffs. Khi inputs thiếu, dừng và báo; không thay bằng synthetic data hoặc tự tạo metrics.

## Chạy từng phần analysis

Chỉ chạy sau khi unified, baseline predictions và các input caches theo source đã sẵn sàng. Những lệnh này tạo output local; không train lại MLP. Không chạy lại vào thư mục có output cũ nếu script từ chối overwrite.

### Volatility analysis

```powershell
& .\.venv\Scripts\python.exe .\reproduction_setup\volatility_validation_analysis.py
```

### Market Regime analysis

```powershell
& .\.venv\Scripts\python.exe .\research_regime_analysis\analyze_validation.py
& .\.venv\Scripts\python.exe .\research_regime_analysis\robustness.py
```

### Autocorrelation analysis

```powershell
& .\.venv\Scripts\python.exe .\research_autocorrelation_analysis\explore.py
& .\.venv\Scripts\python.exe .\research_autocorrelation_analysis\conditional_analysis.py
& .\.venv\Scripts\python.exe .\research_autocorrelation_analysis\summarize.py
```

### Distribution Shift analysis

```powershell
& .\.venv\Scripts\python.exe .\research_distribution_shift\build_scores.py
& .\.venv\Scripts\python.exe .\research_distribution_shift\describe.py
& .\.venv\Scripts\python.exe .\research_distribution_shift\conditional_analysis.py
& .\.venv\Scripts\python.exe .\research_distribution_shift\robustness.py
```

Các `write_report.py` tạo lại Markdown analysis từ outputs local, không cần chạy để đọc report đã có. Chuỗi commands không tự dựng các historical A/reference caches của threshold methods; cần chuẩn bị riêng các inputs được liệt kê bên dưới, không giả định analysis outputs đã đủ cho mọi script.

## Volatility-B

Cần `validation_decisions.parquet`, `test_decisions.parquet`, `locked_config_before_test.json` và metrics tham chiếu trong `research_volatility_threshold/full_validation_test_20261103/`.

Dùng môi trường Python đã chuẩn bị, chạy PowerShell từ repository:

```powershell
Set-Location 'D:\crypto-confidence-execution'
& .\reproduction_setup\venv\Scripts\python.exe -c "import sys; from pathlib import Path; from datetime import datetime; sys.path.insert(0, 'research_volatility_threshold'); import method_b_benchmark as m; m.OUT=Path('research_runs') / ('volatility_b_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')); m.main()"
```

Script không có `--out`; lệnh chỉ đặt biến output `OUT` trong process mới rồi gọi `main()`, không sửa file hoặc hyperparameters. Lệnh fit/chọn threshold bằng validation rồi đánh giá test theo cấu hình script; không train lại MLP. Guard theo stage giữ peak working set≤9GiB và Available RAM≥2GiB.

Output trong `research_runs/`: cấu hình đã khóa, grids, `metrics_validation_test.csv`, decisions validation/test, coverage/weekly diagnostics và resource logs. Những input/model/reference paths khác vẫn lấy theo source, không đổi theo `--out`.


## Regime-B

Cần artifacts Regime-A tại `research_regime_threshold/method_a_20261103/`, model/predictions BTC/ETH và các báo cáo/config gốc được script kiểm tra hash.

Dùng môi trường Python đã chuẩn bị, chạy PowerShell từ repository:

```powershell
Set-Location 'D:\crypto-confidence-execution'
$methodRunOut = 'research_runs/regime_b_' + (Get-Date -Format yyyyMMdd_HHmmss_fff)
& .\reproduction_setup\venv\Scripts\python.exe .\research_regime_threshold\run_regime_b.py --out $methodRunOut
```

`--out` trỏ đến thư mục có timestamp, chưa tồn tại; script từ chối ghi đè thư mục cũ. Lệnh fit/chọn threshold bằng validation rồi đánh giá test theo cấu hình script; không train lại MLP. Guard theo stage giữ peak working set≤9GiB và Available RAM≥2GiB.

Output trong `research_runs/`: cấu hình đã khóa, grids, `metrics_validation_test.csv`, decisions validation/test, coverage/weekly diagnostics và resource logs. Những input/model/reference paths khác vẫn lấy theo source, không đổi theo `--out`.


## ACF-A

Cần unified BTC/ETH, saved predictions trong `data/btc_eth_1pct/training/`, threshold selection, cached ACF/folds trong `research_autocorrelation_analysis/validation_20261103/` và metrics tham chiếu Regime/Volatility.

Dùng môi trường Python đã chuẩn bị, chạy PowerShell từ repository:

```powershell
Set-Location 'D:\crypto-confidence-execution'
$methodRunOut = 'research_runs/acf_a_' + (Get-Date -Format yyyyMMdd_HHmmss_fff)
& .\reproduction_setup\venv\Scripts\python.exe .\research_autocorrelation_threshold\run_acf_a.py --out $methodRunOut
```

`--out` trỏ đến thư mục có timestamp, chưa tồn tại; script từ chối ghi đè thư mục cũ. Lệnh fit/chọn threshold bằng validation rồi đánh giá test theo cấu hình script; không train lại MLP. Guard theo stage giữ peak working set≤9GiB và Available RAM≥2GiB.

Output trong `research_runs/`: cấu hình đã khóa, grids, `metrics_validation_test.csv`, decisions validation/test, coverage/weekly diagnostics và resource logs. Những input/model/reference paths khác vẫn lấy theo source, không đổi theo `--out`.


## Shift-A

Cần matrices/metadata/manifests trong `data/btc_eth_1pct/prepared/`, saved predictions, training PCA reference và cached analysis/folds trong `research_distribution_shift/validation_20261103/`, cùng config/metrics ACF tham chiếu.

Dùng môi trường Python đã chuẩn bị, chạy PowerShell từ repository:

```powershell
Set-Location 'D:\crypto-confidence-execution'
$methodRunOut = 'research_runs/shift_a_' + (Get-Date -Format yyyyMMdd_HHmmss_fff)
& .\reproduction_setup\venv\Scripts\python.exe .\research_distribution_shift_threshold\run_shift_a.py --out $methodRunOut
```

`--out` trỏ đến thư mục có timestamp, chưa tồn tại; script từ chối ghi đè thư mục cũ. Lệnh fit/chọn threshold bằng validation rồi đánh giá test theo cấu hình script; không train lại MLP. Guard theo stage giữ peak working set≤9GiB và Available RAM≥2GiB.

Output trong `research_runs/`: cấu hình đã khóa, grids, `metrics_validation_test.csv`, decisions validation/test, coverage/weekly diagnostics và resource logs. Những input/model/reference paths khác vẫn lấy theo source, không đổi theo `--out`.

## Phạm vi tái lập

Các lệnh method ghi metrics thật từ inputs được cung cấp; không sinh lại bảng kịch bản trong report. Nhiều input paths/schema đang gắn với historical run; đây chưa phải clean-room, one-command reproduction package. Không chạy lại model hoặc sửa methodology chỉ để khớp bảng. Oracle filtering, horizon theo dòng, overlap/missing purge và validation reuse vẫn tồn tại; kết quả trên test đã xem là retrospective/exploratory.

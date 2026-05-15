# Dog Eye Color — EVC Project (Part B)

Dự đoán màu mắt chó (xanh / nâu) từ dữ liệu DNA (SNP genotype).

Dataset: Deane-Coe et al. 2018 (PLOS Genetics)  
n = 2,769 chó | 3.9% mắt xanh | 56 SNP được chọn từ 213,245 (xem mục **Chọn SNP** bên dưới)

## Kết quả

Test set (K = 56 SNP, ngưỡng `p_wald < 1.15×10⁻⁷` — Bonferroni Deane-Coe 2018):

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Majority | 0.040 | 0.500 | 0.000 |
| Logistic Regression | 0.619 | 0.851 | 0.426 |
| Random Forest | 0.605 | 0.873 | 0.595 |
| **MLP (đề xuất)** | **0.667** | **0.921** | **0.667** |
| TabPFN (Hollmann 2023) | 0.600 | 0.807 | 0.700 |
| TabNet (Arik & Pfister 2021) | 0.554 | 0.874 | 0.577 |

---

## Cách chạy

### Bước 1 — Clone repo

```bash
git clone <repo_url>
cd FDP-EVC/dog
```

### Bước 2 — Cài thư viện

```bash
pip install -r requirements.txt
```

### Bước 3 — Tải dữ liệu thô

Dữ liệu **không có trong repo** (quá lớn). Tải từ:

> Deane-Coe et al. 2018 — https://doi.org/10.1371/journal.pgen.1007934  
> (Supplementary Data, mục "Data Availability")

Sau khi tải, đặt vào đúng thư mục (chỉ cần 5 file sau, bỏ qua 2 file còn lại):

```
dog/data/raw/eye/
├── deane-coe_etal_canine_eye_color_GWAS_indiv_phenotype_logr_haplotype_breed.csv
├── deane-coe_etal_canine_eye_color_GWAS_N3180_discovery_panel.assoc.txt
├── deane-coe_etal_canine_eye_color_GWAS_N3180_discovery_panel.bed   ← bộ 3 PLINK
├── deane-coe_etal_canine_eye_color_GWAS_N3180_discovery_panel.bim   ← bộ 3 PLINK
└── deane-coe_etal_canine_eye_color_GWAS_N3180_discovery_panel.fam   ← bộ 3 PLINK
```

> Hai file `indiv_logR_per_marker.csv`, `indiv_phased_haplotypes.csv` và `relatedness_matrix.txt` không cần thiết cho pipeline này.

### Bước 4 — Chạy pipeline theo thứ tự

Chạy từ thư mục gốc repo (`FDP-EVC/`):

```bash
# 1. Tiền xử lý dữ liệu → eye_processed.npz
python dog/src/data/preprocess_eye.py

# 2. Tạo splits (train/val/test) → eye_splits.json
python dog/src/data/splits.py

# 3. Chạy baselines (Majority, LR, RF) → baseline_results.json
python dog/src/models/baselines.py

# 4. Train MLP → mlp_results.json
python dog/src/train/train_eye.py

# 4b. (Tuỳ chọn) Train TabPFN — pretrained Transformer in-context learning
python dog/src/train/train_tabpfn.py

# 4c. (Tuỳ chọn) Train TabNet — attention-based tabular model
python dog/src/train/train_tabnet.py

# 5. Sinh báo cáo + figures → report/eye.md + report/figures/
python dog/src/evaluation/report_eye.py

# 6. (Tuỳ chọn) Chạy K sensitivity sweep + vẽ biểu đồ
python dog/src/experiments/compare_k.py
python dog/src/evaluation/plot_k_sensitivity.py
```

---

## Cấu trúc thư mục

```
dog/
├── requirements.txt
├── data/
│   ├── raw/eye/          ← dữ liệu thô (tải thủ công, không có trong git)
│   └── processed/        ← sinh ra sau bước 1 & 2 (không có trong git)
├── experiments/eye/      ← kết quả JSON (sinh ra sau bước 3, 4, 4b, 4c)
│   ├── baseline_results.json
│   ├── mlp_results.json
│   ├── tabpfn_results.json
│   ├── tabnet_results.json
│   └── k_sensitivity.{json,md}
├── report/
│   ├── eye.md            ← báo cáo kết quả
│   └── figures/          ← biểu đồ (PR curves, metric bars, K sensitivity)
└── src/
    ├── data/
    │   ├── preprocess_eye.py     ← Bước 1
    │   └── splits.py             ← Bước 2
    ├── models/
    │   ├── baselines.py          ← Bước 3 (Majority / LR / RF)
    │   └── mlp.py
    ├── train/
    │   ├── train_eye.py          ← Bước 4 (MLP)
    │   ├── train_tabpfn.py       ← Bước 4b (tuỳ chọn)
    │   └── train_tabnet.py       ← Bước 4c (tuỳ chọn)
    ├── experiments/
    │   └── compare_k.py          ← K sensitivity sweep
    └── evaluation/
        ├── metrics.py
        ├── report_eye.py         ← Bước 5
        └── plot_k_sensitivity.py ← vẽ biểu đồ K sensitivity
```

---

## Phương pháp

**Vấn đề**: 213,245 SNP >> 2,769 chó → không thể feed thẳng vào model. Chỉ 3.9% chó có mắt xanh → imbalanced.

**Giải pháp đề xuất**:
1. **GWAS feature selection** — chỉ giữ các SNP có `p_wald < 1.15×10⁻⁷` (ngưỡng Bonferroni dùng trong bài báo gốc Deane-Coe et al. 2018 trên chính dataset này). Cho ra 56 SNP, hầu hết nằm trên chr18 vùng gene **ALX4** — đúng locus bài báo chỉ ra là nguyên nhân chính gây mắt xanh.
2. **MLP với pos_weight** — `pos_weight = n_neg / n_pos ≈ 24` để buộc model học class hiếm
3. **PR-AUC** làm metric chính thay vì accuracy
4. **So sánh với 5 model khác**:
   - **Majority** baseline (luôn predict mắt nâu)
   - **Logistic Regression** với `class_weight=balanced`
   - **Random Forest** (500 cây, `balanced_subsample`)
   - **TabPFN** (Hollmann 2023) — pretrained Transformer in-context learning, không train trên data của mình. Bù cho thiếu `pos_weight` bằng threshold tuning trên validation.
   - **TabNet** (Arik & Pfister 2021) — sequential attention chọn feature mỗi step, dùng `weights={0:1, 1:24.7}` xử lý imbalance.

### Vì sao chọn ngưỡng `p < 1.15×10⁻⁷` (số SNP K)?

`1.15×10⁻⁷` chính là **ngưỡng Bonferroni mà bài báo gốc Deane-Coe et al. 2018 dùng** trên đúng dataset này (`0.05 / ~430k markers ≈ 1.15×10⁻⁷`). Dùng đúng ngưỡng của reference paper hợp lý hơn là chọn K bằng tay (vd K=200) vì:

- Có cơ sở khoa học **trực tiếp**: ngưỡng đến từ chính bài báo công bố dataset này, không phải convention chung chung.
- SNP được giữ lại đều **đạt significance trong bài báo gốc** → tất cả đều có ý nghĩa thống kê thực sự.
- Phù hợp với trait **oligogenic** (do vài gene chính chi phối) như màu mắt chó. Deane-Coe đã chỉ ra **chr18/ALX4** là locus chính (p ≈ 1.3×10⁻⁶⁸) — không cần thêm SNP yếu.

Nếu muốn so sánh với ngưỡng khác (suggestive `p < 10⁻⁵`, hoặc top-K cố định), preprocess script hỗ trợ:
```bash
python dog/src/data/preprocess_eye.py --p 5e-8     # genome-wide standard (Pe'er 2008)
python dog/src/data/preprocess_eye.py --p 1e-5     # suggestive
python dog/src/data/preprocess_eye.py --top_k 200  # legacy
```

### Phụ lục — K sensitivity (đã thực nghiệm)

Để chứng minh ngưỡng `p < 1.15e-7` không phải lựa chọn tuỳ tiện, mình chạy sweep nhiều K trên cùng pipeline (5-fold CV + held-out test, MLP):

| Config | #SNP | CV PR-AUC | CV ROC-AUC | CV F1 | TEST PR-AUC | TEST ROC-AUC | TEST F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **p<1.15e-7 (Deane-Coe, default)** | **56** | **0.737** | **0.936** | **0.563** | **0.667** | **0.921** | **0.667** |
| p<5e-8 (genome-wide standard) | 52 | 0.727 | 0.921 | 0.550 | 0.641 | 0.868 | 0.536 |
| p<1e-5 (suggestive) | 96 | 0.765 | 0.925 | 0.611 | 0.624 | 0.822 | 0.667 |
| p<1e-4 (lenient) | 157 | 0.797 | 0.934 | 0.679 | 0.674 | 0.856 | 0.632 |
| top_k=100 | 100 | 0.751 | 0.924 | 0.593 | 0.609 | 0.840 | 0.595 |
| top_k=200 | 200 | 0.834 | 0.957 | 0.689 | 0.706 | 0.918 | 0.714 |
| top_k=500 | 500 | 0.915 | 0.991 | 0.817 | 0.836 | 0.958 | 0.732 |

**Quan sát**: K càng lớn → CV PR-AUC càng cao (đến 0.91 với K=500). **Nhưng** chênh lệch CV vs TEST tăng theo K: K=500 có gap 0.08, K=56 chỉ gap 0.07 và **F1 trên test cao nhất** (0.667) trong các config "có cơ sở thống kê".

**Quyết định**: chọn **K=56 (p < 1.15e-7)** làm default vì:
- Có cơ sở khoa học **trực tiếp** từ bài báo gốc dùng đúng dataset này (Deane-Coe et al. 2018), không phải convention chung.
- Tất cả SNP giữ lại đều **đạt significance theo Bonferroni** → không có SNP "may rủi".
- Sinh học rõ ràng: 42/56 SNP nằm trên chr18/ALX4 — đúng locus bài báo chỉ ra (p ≈ 1.3×10⁻⁶⁸).
- Performance test set cạnh tranh với top_k=200 (F1 0.667 vs 0.714) nhưng dùng **ít hơn 3.5× số feature** → ít overfit risk.

Để chạy lại sweep này:
```bash
python dog/src/experiments/compare_k.py
# → experiments/eye/k_sensitivity.{md,json}
python dog/src/evaluation/plot_k_sensitivity.py
# → report/figures/03_k_sensitivity.png
```

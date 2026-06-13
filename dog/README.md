# Dog Eye Color — EVC Project (Part B)

Dự đoán màu mắt chó (xanh / nâu) từ dữ liệu DNA (SNP genotype).

Dataset: Deane-Coe et al. 2018 (PLOS Genetics)  
n = 2,769 chó | 3.9% mắt xanh | 52 SNP được chọn từ 213,245 bằng ngưỡng paper `p_wald < 5e-8` (xem mục **Chọn SNP** bên dưới)

## Kết quả

Test set (K = 52 SNP, ngưỡng genome-wide significance `p_wald < 5×10⁻⁸` trong Deane-Coe et al. 2018):

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Majority | 0.040 | 0.500 | 0.000 |
| Logistic Regression | 0.625 | 0.854 | 0.455 |
| Random Forest | 0.628 | 0.853 | 0.556 |
| **MLP (đề xuất)** | **0.641** | **0.868** | **0.536** |
| MLP (tuned) | 0.670 | 0.891 | 0.652 |
| TabPFN (Hollmann 2023) | 0.600 | 0.836 | 0.667 |
| TabICL (Qu 2025) | 0.673 | 0.876 | 0.727 |
| TabNet (Arik & Pfister 2021) | 0.549 | 0.817 | 0.262 |

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

# 4c. (Tuỳ chọn) Train TabICL — tabular foundation model in-context learning
python dog/src/train/train_tabicl.py

# 4d. (Tuỳ chọn) Train TabNet — attention-based tabular model
python dog/src/train/train_tabnet.py

# 5. Sinh báo cáo + figures → report/eye.md + report/figures/
python dog/src/evaluation/report_eye.py
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
│   ├── tabicl_results.json
│   └── tabnet_results.json
├── report/
│   ├── eye.md            ← báo cáo kết quả
│   └── figures/          ← biểu đồ (PR curves, metric bars)
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
    │   ├── train_tabicl.py       ← Bước 4c (tuỳ chọn)
    │   └── train_tabnet.py       ← Bước 4d (tuỳ chọn)
    └── evaluation/
        ├── metrics.py
        └── report_eye.py         ← Bước 5
```

---

## Phương pháp

**Vấn đề**: 213,245 SNP >> 2,769 chó → không thể feed thẳng vào model. Chỉ 3.9% chó có mắt xanh → imbalanced.

**Giải pháp đề xuất**:
1. **GWAS feature selection** — chỉ giữ các SNP có `p_wald < 5×10⁻⁸`, đúng ngưỡng genome-wide significance được Deane-Coe et al. 2018 ghi trong paper. Cho ra 52 SNP, hầu hết nằm trên chr18 vùng gene **ALX4** — đúng locus bài báo chỉ ra là nguyên nhân chính gây mắt xanh.
2. **MLP với pos_weight** — `pos_weight = n_neg / n_pos ≈ 24` để buộc model học class hiếm
3. **PR-AUC** làm metric chính thay vì accuracy
4. **So sánh với các model đã rerun trên bundle 52 SNP**:
   - **Majority** baseline (luôn predict mắt nâu)
   - **Logistic Regression** với `class_weight=balanced`
   - **Random Forest** (500 cây, `balanced_subsample`)
   - **MLP (tuned)** — grid search hyperparameter trên trainval, chọn bằng CV PR-AUC.
   - **TabPFN** (Hollmann 2023) — pretrained Transformer in-context learning, bù cho thiếu `pos_weight` bằng threshold tuning trên validation.
   - **TabICL** (Qu 2025) — tabular foundation model in-context learning, xử lý bảng qua column-wise embedding, row-wise interaction và dataset-wise ICL. Bù cho thiếu `pos_weight` bằng threshold tuning trên validation.
   - **TabNet** (Arik & Pfister 2021) — sequential attention chọn feature mỗi step, dùng `weights={0:1, 1:24.7}` xử lý imbalance.

### Vì sao chọn ngưỡng `p < 5×10⁻⁸` (số SNP K)?

`5×10⁻⁸` là **ngưỡng genome-wide significance được paper Deane-Coe et al. 2018 ghi trực tiếp**. Dùng đúng ngưỡng paper hợp lý hơn là chọn K bằng tay (vd K=200) vì:

- Có cơ sở khoa học **trực tiếp**: ngưỡng đến từ chính bài báo công bố dataset này.
- SNP được giữ lại đều **đạt significance trong bài báo gốc** → tất cả đều có ý nghĩa thống kê thực sự.
- Phù hợp với trait **oligogenic** (do vài gene chính chi phối) như màu mắt chó. Deane-Coe đã chỉ ra **chr18/ALX4** là locus chính (p ≈ 1.3×10⁻⁶⁸) — không cần thêm SNP yếu.

Nếu muốn so sánh với ngưỡng khác (suggestive `p < 10⁻⁵`, hoặc top-K cố định), preprocess script hỗ trợ:
```bash
python dog/src/data/preprocess_eye.py --p 5e-8     # ngưỡng paper Deane-Coe 2018
python dog/src/data/preprocess_eye.py --p 1e-5     # suggestive
python dog/src/data/preprocess_eye.py --top_k 200  # legacy
```

**Quyết định hiện tại**: chọn **K=52 (p < 5e-8)** làm default vì:
- Khớp số liệu/ngưỡng được ghi trực tiếp trong paper Deane-Coe et al. 2018.
- Tất cả SNP giữ lại đều đạt genome-wide significance theo paper.
- Sinh học rõ ràng: phần lớn SNP nằm trên chr18/ALX4 — đúng locus bài báo chỉ ra (p ≈ 1.3×10⁻⁶⁸).
- Tránh claim sai rằng `1.15e-7` là threshold của paper.

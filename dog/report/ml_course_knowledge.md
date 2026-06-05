# IT3190 Machine Learning - Knowledge Notes

File này tóm tắt kiến thức từ các slide:

- `L1-Intro-HauLD-Shared.pdf`
- `L2-Data-crawling-preprocessing.pdf`
- `L3-linear-regression-HauLD.pdf`
- `L4-5-Clustering-HauLD.pdf`
- `L6-Random-forests-HauLD.pdf`
- `L7-Neural-network-HauLD.pdf`
- `L8-SVM-HauLD.pdf`

Mục tiêu của file này không chỉ là chép lại slide, mà là giải thích lại theo kiểu người mới học có thể đọc được:

- Công thức dùng ký hiệu rõ ràng.
- Thuật toán được viết thành từng bước.
- Mỗi model có ví dụ trực giác.
- Câu hỏi sau này sẽ được thêm vào đúng lesson tương ứng.

## Cách đọc file này

Nếu cần ôn nhanh:

```text
L1  -> ML là gì, supervised/unsupervised, classification/regression
L2  -> dữ liệu được crawl, làm sạch, chuẩn hóa, biến thành feature như thế nào
L3  -> linear regression, OLS, Ridge, LASSO
L4-5 -> clustering, K-means, K-means++, hierarchical clustering, evaluation
L6  -> decision tree, ID3, entropy, information gain, random forest
L7  -> neural network, perceptron, gradient descent, backpropagation
L8  -> SVM, margin, hard/soft margin, dual form, kernel trick
```

Nếu cần hiểu công thức, đọc chậm các phần:

```text
L3.4  OLS derivation
L3.5  Ridge
L3.6  LASSO
L4.4  K-means objective
L6.4  Entropy and information gain
L7.7  Backpropagation
L8.4  Hard-margin SVM
L8.6  Dual form
L8.8  Kernel trick
```

## Quy ước cập nhật Q&A

Từ sau khi file này được tạo, nếu có câu hỏi về một lesson cụ thể, câu trả lời ngắn sẽ được thêm vào mục `Q&A log` của lesson đó.

Ví dụ:

```text
Câu hỏi về entropy -> thêm vào L6 Q&A log
Câu hỏi về gradient descent -> thêm vào L7 Q&A log
Câu hỏi về kernel SVM -> thêm vào L8 Q&A log
```

Nếu câu hỏi liên quan nhiều lesson, nó sẽ được đưa vào lesson chính và có cross-reference.

# Lesson 1. Introduction to Machine Learning and Data Mining

## L1.1. Machine Learning là gì?

Machine Learning là cách xây dựng hệ thống có thể cải thiện hiệu năng nhờ học từ dữ liệu.

Một định nghĩa thực dụng:

```text
Ta có một task T, một performance measure P, và experience/data E.
Nếu hệ thống cải thiện P trên T nhờ E, ta nói hệ thống đang học.
```

Trong bài toán dog eye color của project:

```text
T = dự đoán chó có mắt xanh hay không
P = PR-AUC, ROC-AUC, F1
E = dữ liệu SNP + label eye color của các dogs
```

Điểm quan trọng: ML không phải là viết rule thủ công kiểu:

```text
if SNP_123 == 1 then blue
```

mà là học một hàm:

```text
f(x) -> y
```

trong đó:

- `x` là input/features.
- `y` là output/label.
- `f` là model.

## L1.2. Data Mining là gì?

Data Mining rộng hơn một model đơn lẻ. Nó là quá trình khai phá tri thức từ dữ liệu.

Pipeline thường thấy:

```text
Data collection
    ↓
Data processing / preprocessing
    ↓
Data visualization / grasping
    ↓
Analysis / hypothesis testing / machine learning
    ↓
Insight / policy decision
```

Machine Learning thường nằm ở bước phân tích/học mô hình, còn Data Mining quan tâm cả quá trình từ lấy dữ liệu đến tạo insight.

## L1.3. Supervised learning

Supervised learning là học có giám sát: dữ liệu train có cả input và label.

Ký hiệu:

```text
D = {(x_1, y_1), (x_2, y_2), ..., (x_M, y_M)}
```

Trong đó:

- `x_i` là vector feature của sample thứ `i`.
- `y_i` là label/response của sample thứ `i`.
- `M` là số sample.

Mục tiêu:

```text
Tìm hàm f sao cho f(x_i) ≈ y_i
và quan trọng hơn: f dự đoán tốt trên dữ liệu mới.
```

Có hai nhánh lớn:

```text
Classification:
  y thuộc tập rời rạc.
  Ví dụ: y ∈ {brown, blue}

Regression:
  y là số thực.
  Ví dụ: dự đoán giá nhà, stock index, chiều cao.
```

### Classification

Ví dụ binary classification:

```text
x = SNP vector của một dog
y = 1 nếu blue eye, 0 nếu brown/non-blue
```

Model xuất có thể là:

```text
hard label: blue / brown
probability: P(blue | x) = 0.73
```

Với class imbalance, probability thường hữu ích hơn hard label vì có thể tune threshold:

```text
predict blue nếu P(blue) >= threshold
```

### Regression

Ví dụ:

```text
x = [diện tích nhà, số phòng, vị trí]
y = giá nhà
```

Model cần dự đoán một số thực:

```text
f(x) = 3.2 tỷ
```

## L1.4. Unsupervised learning

Unsupervised learning là học không giám sát: dữ liệu train chỉ có input, không có label.

Ký hiệu:

```text
D = {x_1, x_2, ..., x_M}
```

Mục tiêu không phải dự đoán label có sẵn, mà là tìm cấu trúc ẩn:

```text
clusters
patterns
trends
low-dimensional representation
```

Ví dụ:

```text
Input: dữ liệu khách hàng không có nhãn
Output: nhóm khách hàng tương tự nhau
```

## L1.5. Generalization

Một model tốt không chỉ fit dữ liệu train. Nó phải dự đoán tốt dữ liệu chưa thấy.

```text
training performance tốt nhưng test performance kém
    -> overfitting

training performance kém
    -> underfitting
```

Nền tảng của các lesson sau:

```text
L3: regularization Ridge/LASSO để giảm overfitting
L6: pruning / random forest để giảm overfitting của cây
L7: validation, learning rate, architecture để train NN tốt hơn
L8: margin và C trong SVM để cân bằng fit/generalization
```

## L1 Q&A log

Chưa có câu hỏi riêng sau khi tạo file.

### Q1. Machine learning có phải là dạy cho máy làm việc giống con người không?

Không hẳn. Machine learning là dạy máy học quy luật từ dữ liệu để làm tốt một task nào đó, không bắt buộc phải suy nghĩ hoặc làm việc giống con người. Có những bài toán nhìn giống năng lực con người, như nhận diện ảnh, dịch ngôn ngữ, phân loại spam. Nhưng bản chất ML thường là tối ưu một hàm dự đoán `f(x) -> y` dựa trên dữ liệu và loss function. Ví dụ trong dog eye color project, ta không dạy máy "nhìn chó như người"; ta cho model dữ liệu SNP và label mắt xanh/mắt nâu, rồi model học pattern SNP nào liên quan tới `P(blue eye)`.

# Lesson 2. Data Crawling and Preprocessing

## L2.1. Vì sao phải preprocessing?

Model ML thường không làm việc trực tiếp với raw data như HTML, text tự do, ảnh thô hoặc dữ liệu lỗi.

Model thường cần dữ liệu có cấu trúc:

```text
matrix X: shape = (n_samples, n_features)
vector y: shape = (n_samples,)
```

Ví dụ với dog project:

```text
X[i, j] = genotype value của dog i tại SNP j
y[i]    = eye color label của dog i
```

Preprocessing giúp:

- Dữ liệu lưu trữ/truy vấn thuận tiện.
- Feature có representation phù hợp.
- Giảm noise/lỗi.
- Đưa nhiều loại dữ liệu về dạng số.
- Tránh model bị lệch vì scale feature khác nhau.

## L2.2. Data collection

Các cách thu thập dữ liệu trong slide:

```text
Sampling  -> lấy mẫu từ một population
Crawling  -> tự động đi qua các link/trang để lấy dữ liệu
Logging   -> ghi lại hành vi/sự kiện
Scraping  -> bóc nội dung từ web/page
```

Điểm quan trọng: dữ liệu train là một mẫu nhỏ của bài toán thật, nên nó cần phản ánh đủ các khía cạnh quan trọng.

Nếu sample bị lệch:

```text
train data chủ yếu là brown-eye dogs
    -> model dễ học bias "luôn brown"
```

## L2.3. Data quality

Slide nhấn mạnh dữ liệu trước khi vào ML nên có các tính chất:

```text
completeness  -> đủ thông tin cần thiết
integrity     -> không mâu thuẫn/lỗi logic
homogeneity   -> cùng format/kiểu biểu diễn
well-defined structure -> cấu trúc rõ ràng
```

Ví dụ lỗi:

```text
Dog A label = blue ở file 1
Dog A label = brown ở file 2
```

Đây là lỗi integrity.

## L2.4. Cleaning và missing values

Missing value:

```text
x_ij = ?
```

Cách xử lý phổ biến:

```text
Numeric feature:
  fill mean/median

Categorical feature:
  fill mode

Model/tree-specific:
  có thể dùng nhánh missing riêng hoặc surrogate split
```

Công thức mean imputation:

```text
mean_j = average của feature j trên train set
x_ij' = mean_j nếu x_ij missing
```

Lưu ý cực quan trọng:

```text
mean_j phải tính trên train set, không tính trên toàn bộ train+test.
```

Nếu tính trên test, ta làm data leakage.

## L2.5. Feature extraction

Raw data phải được biến thành feature.

Ví dụ text:

```text
raw text:
  "the dog has blue eyes"

bag-of-words:
  dog: 1
  blue: 1
  eyes: 1
```

Ví dụ genotype:

```text
AA -> 0
Aa -> 1
aa -> 2
```

Sau đó model mới nhận vector số:

```text
x = [0, 1, 0, 2, ...]
```

## L2.6. Discretization

Discretization là rời rạc hóa feature liên tục.

Ví dụ:

```text
age ∈ [0, 100]
```

Chia thành:

```text
[0, 18)    -> young
[18, 60)   -> adult
[60, 100]  -> senior
```

Với decision tree cổ điển như ID3, discretization giúp biến real-valued attributes thành categorical attributes để split dễ hơn.

## L2.7. Normalization và standardization

### Min-max normalization

Đưa feature về `[0, 1]`:

```text
x' = (x - min) / (max - min)
```

Ví dụ:

```text
feature weight có min=2, max=10
x=6
x'=(6-2)/(10-2)=0.5
```

### Z-score standardization

Đưa feature về mean 0, std 1:

```text
x' = (x - μ) / σ
```

Trong đó:

- `μ` là mean của feature.
- `σ` là standard deviation.

Trong project, các model như MLP/TabPFN/TabICL dùng standardize:

```text
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
X_train' = (X_train - mean) / std
X_test'  = (X_test  - mean) / std
```

Điểm mấu chốt:

```text
Không dùng test set để tính mean/std.
```

## L2.8. One-hot encoding

Categorical feature:

```text
Outlook ∈ {Sunny, Rain, Overcast}
```

One-hot:

```text
Sunny    -> [1, 0, 0]
Rain     -> [0, 1, 0]
Overcast -> [0, 0, 1]
```

Nhiều model cần one-hot vì không hiểu string label trực tiếp.

## L2 Q&A log

Chưa có câu hỏi riêng sau khi tạo file.

### Q1. Vì sao SNP có thể chuyển về các số 0, 1, 2? Quy luật là gì?

SNP là một vị trí trên DNA nơi các cá thể có thể khác nhau ở một nucleotide, ví dụ allele `A` hoặc `G`. Vì chó/người thường có 2 bản sao chromosome ở mỗi vị trí autosome, mỗi cá thể có 2 allele tại một SNP. Nếu SNP là biallelic, tức chỉ xét 2 allele như `A/G`, genotype có 3 dạng chính: `AA`, `AG`, `GG`. Ta chọn một allele để đếm, thường là alternate allele hoặc minor allele. Nếu đếm allele `G`, quy luật mã hóa là `AA -> 0` vì có 0 allele `G`, `AG -> 1` vì có 1 allele `G`, `GG -> 2` vì có 2 allele `G`. Vì vậy số `0/1/2` không phải 3 loại SNP khác nhau; nó là số lượng allele được chọn trong genotype của sample tại SNP đó.

### Q2. Làm sao biết chọn allele nào để đếm? Hai allele ở một SNP có phải do DNA có hai đoạn xoắn vào nhau không?

Allele được chọn để đếm phụ thuộc vào quy ước của dữ liệu. Nếu dữ liệu ở dạng VCF, mỗi SNP có `REF` và `ALT`; dosage thường đếm `ALT`, nên `0/0 -> 0`, `0/1 -> 1`, `1/1 -> 2`. Nếu dữ liệu ở dạng PLINK/GWAS, file metadata có thể ghi allele 1/allele 2, minor allele, hoặc effect allele; preprocessing phải ghi rõ đang đếm allele nào. Điều quan trọng nhất là phải nhất quán trên toàn dataset. Nếu đổi allele được đếm, mã hóa sẽ đảo `x -> 2 - x`, ví dụ đếm `G` thì `AA=0, AG=1, GG=2`, còn đếm `A` thì `AA=2, AG=1, GG=0`.

Hai allele ở một SNP không phải do hai sợi trong một DNA double helix. Một phân tử DNA đúng là có hai sợi xoắn kép, nhưng hai sợi đó bổ sung nhau, ví dụ một sợi có `A` thì sợi kia là `T`; đó không phải hai allele độc lập. Số `0/1/2` xuất hiện vì chó/người là sinh vật diploid: tại phần lớn vị trí autosome, cá thể có hai chromosome tương đồng, một bản sao từ bố và một bản sao từ mẹ. Mỗi bản sao có một allele tại cùng vị trí SNP. Ví dụ bản từ bố có `A`, bản từ mẹ có `G` thì genotype là `AG`, và nếu đang đếm allele `G` thì dosage bằng `1`.

# Lesson 3. Linear Regression

## L3.1. Bài toán regression

Ta có dataset:

```text
D = {(x_1, y_1), ..., (x_M, y_M)}
```

Trong đó:

```text
x_i = (x_i1, x_i2, ..., x_in)^T
y_i ∈ R
```

Mục tiêu:

```text
Học f sao cho y_i ≈ f(x_i)
```

Với linear regression, giả sử:

```text
f(x, w) = w_0 + w_1 x_1 + ... + w_n x_n
```

`w_0` là bias/intercept.

## L3.2. Matrix form

Thêm cột 1 vào mỗi sample để chứa bias:

```text
a_i = [1, x_i1, x_i2, ..., x_in]
```

Tạo matrix:

```text
A =
[
  [1, x_11, x_12, ..., x_1n],
  [1, x_21, x_22, ..., x_2n],
  ...
  [1, x_M1, x_M2, ..., x_Mn]
]

w = [w_0, w_1, ..., w_n]^T
y = [y_1, y_2, ..., y_M]^T
```

Prediction toàn bộ train set:

```text
y_hat = A w
```

Residual:

```text
r = y - A w
```

## L3.3. Loss, risk, empirical loss

Với một sample:

```text
loss(f, x) = (y - f(x, w))^2
```

Expected loss/risk:

```text
R(f) = E_x[(f*(x) - f(x, w))^2]
```

Nhưng ta không biết phân phối thật của mọi `x`, nên không tối ưu trực tiếp được `R(f)`.

Ta dùng empirical loss trên dataset:

```text
RSS(w) = Σ_i (y_i - f(x_i, w))^2
       = ||y - A w||_2^2
```

RSS là residual sum of squares.

## L3.4. Ordinary Least Squares - OLS

OLS chọn `w` để RSS nhỏ nhất:

```text
w* = argmin_w ||y - A w||_2^2
```

Mở rộng:

```text
J(w) = (y - A w)^T (y - A w)
     = y^T y - 2 w^T A^T y + w^T A^T A w
```

Gradient theo `w`:

```text
∇J(w) = -2 A^T y + 2 A^T A w
      = 2 A^T (A w - y)
```

Tại optimum, gradient bằng 0:

```text
2 A^T (A w - y) = 0
A^T A w = A^T y
```

Nếu `A^T A` khả nghịch:

```text
w* = (A^T A)^(-1) A^T y
```

Đây là normal equation.

### OLS algorithm

```text
Input:
  A: data matrix đã thêm cột bias
  y: target vector

Steps:
  1. Tính A^T A
  2. Tính A^T y
  3. Giải hệ (A^T A) w = A^T y
  4. Dùng w để predict x mới

Prediction:
  y_hat = w_0 + w_1 x_1 + ... + w_n x_n
```

### OLS limitations

OLS có vấn đề khi:

```text
A^T A không khả nghịch
```

Điều này xảy ra nếu feature phụ thuộc tuyến tính:

```text
x_3 = 2*x_1 + x_2
```

Hoặc số feature lớn, dữ liệu ít, matrix gần singular.

OLS cũng dễ overfit vì chỉ cố giảm training error, không phạt model quá phức tạp.

## L3.5. Ridge Regression

Ridge thêm L2 regularization:

```text
J_ridge(w) = ||y - A w||_2^2 + λ ||w||_2^2
```

Trong đó:

```text
||w||_2^2 = Σ_j w_j^2
λ > 0
```

Ý nghĩa:

- RSS bắt model fit data.
- `λ ||w||_2^2` phạt weights lớn.
- `λ` điều khiển trade-off fit/generalization.

Gradient:

```text
∇J_ridge(w) = 2 A^T(Aw - y) + 2λw
```

Set gradient = 0:

```text
A^T A w - A^T y + λw = 0
(A^T A + λI)w = A^T y
```

Nghiệm:

```text
w* = (A^T A + λI)^(-1) A^T y
```

Vì `λI` làm matrix ổn định hơn, Ridge thường tránh được singularity của OLS.

### Ridge intuition

Nếu `λ` nhỏ:

```text
Ridge gần giống OLS
fit train tốt hơn nhưng dễ overfit hơn
```

Nếu `λ` lớn:

```text
weights bị kéo nhỏ
model đơn giản hơn
train error có thể tăng
test/generalization có thể tốt hơn
```

## L3.6. LASSO

LASSO dùng L1 regularization:

```text
J_lasso(w) = ||y - A w||_2^2 + λ ||w||_1
```

Trong đó:

```text
||w||_1 = Σ_j |w_j|
```

Điểm khác Ridge:

```text
Ridge/L2  -> kéo weights nhỏ nhưng hiếm khi đúng bằng 0.
LASSO/L1  -> có thể đẩy nhiều weights đúng bằng 0.
```

Vì vậy LASSO có khả năng feature selection:

```text
w_j = 0
    -> feature j bị loại khỏi model
```

LASSO khó tối ưu hơn Ridge vì `|w|` không khả vi tại `w=0`.

## L3.7. OLS vs Ridge vs LASSO

| Method | Objective | Ưu điểm | Nhược điểm |
|---|---|---|---|
| OLS | `RSS` | Đơn giản, nghiệm đóng | Cần inverse, dễ overfit, lỗi nếu singular |
| Ridge | `RSS + λ||w||_2^2` | Ổn định, giảm overfit | Không chọn feature rõ ràng |
| LASSO | `RSS + λ||w||_1` | Có thể tạo sparse weights | Tối ưu khó hơn, nhạy với correlated features |

Một câu nhớ nhanh:

```text
OLS   = fit training data tốt nhất theo square loss.
Ridge = fit data nhưng phạt weights lớn.
LASSO = fit data, phạt weights, và có thể loại feature.
```

## L3 Q&A log

Chưa có câu hỏi riêng sau khi tạo file.

### Q1. Loss, risk, empirical loss và RSS khác nhau như thế nào?

`loss` là lỗi trên một sample cụ thể. Với regression, nếu label thật là `y` và model dự đoán `f(x,w)`, square loss là `(y - f(x,w))^2`. Ta bình phương để lỗi âm/dương đều thành dương và lỗi lớn bị phạt mạnh hơn. `risk` hay expected loss là lỗi trung bình kỳ vọng nếu model được đem dùng trên toàn bộ phân phối dữ liệu thật ngoài đời: `R(f) = E[(f*(x) - f(x,w))^2]`. Đây là thứ ta thật sự muốn nhỏ, vì nó đo khả năng dự đoán trên dữ liệu tương lai. Nhưng ta không biết phân phối thật của mọi `x`, cũng không biết hàm thật `f*(x)`, nên không tối ưu trực tiếp `R(f)` được. Vì vậy ta dùng `empirical loss`, tức lỗi đo trên dataset hữu hạn đã quan sát. Với linear regression, empirical loss thường là RSS: `RSS(w) = Σ_i (y_i - f(x_i,w))^2 = ||y - Aw||_2^2`. RSS là tổng bình phương residuals trên train data; OLS chọn `w` làm RSS nhỏ nhất. Nói ngắn gọn: `loss` = lỗi 1 điểm, `risk` = lỗi kỳ vọng ngoài đời, `empirical loss/RSS` = lỗi đo được trên data train để xấp xỉ risk.

### Q2. L1 và L2 là gì?

L1 và L2 là hai cách đo độ lớn của một vector. Với vector `w = [w1, w2, ..., wn]`, L1 norm là `||w||_1 = |w1| + |w2| + ... + |wn|`, còn L2 norm là `||w||_2 = sqrt(w1^2 + w2^2 + ... + wn^2)`. Trong regularization, người ta thường dùng `||w||_2^2 = Σ_j w_j^2` cho Ridge và `||w||_1 = Σ_j |w_j|` cho LASSO. L2 phạt weight lớn rất mạnh vì có bình phương, nên kéo tất cả weights nhỏ lại nhưng hiếm khi làm weight đúng bằng 0. L1 phạt theo trị tuyệt đối, có hình học tạo nghiệm chạm trục tọa độ, nên nhiều weights có thể trở thành đúng 0; vì vậy LASSO vừa shrinkage vừa feature selection. Nói ngắn gọn: L2 giúp model ổn định và giảm overfitting; L1 giúp model sparse và loại bớt feature.

### Q3. Hình LASSO regularization role giải thích điều gì?

Hình đó vẽ không gian của hai weight `w1` và `w2`. Mỗi điểm trên mặt phẳng là một model khác nhau, ví dụ `(w1=1, w2=0)` nghĩa là feature 1 được dùng còn feature 2 bị loại. Miền màu xanh là tập weight được phép sau regularization. Với L1/LASSO, ràng buộc `|w1| + |w2| <= t` tạo thành hình thoi có các góc nằm đúng trên trục `w1=0` hoặc `w2=0`. Khi đường loss chạm miền được phép, nó rất dễ chạm vào một góc; tại góc đó một weight bằng 0, nên LASSO tạo sparse solution và làm feature selection. Với L2/Ridge, ràng buộc `w1^2 + w2^2 <= t` tạo thành hình tròn, biên trơn không có góc trên trục; điểm chạm thường nằm lệch trục, nên cả `w1` và `w2` nhỏ đi nhưng hiếm khi bằng 0. Vì vậy hình muốn nói: L1 vừa shrink weights vừa chọn feature, còn L2 chủ yếu shrink weights.

### Q4. Vì sao với L2/Ridge đường loss thường chạm hình tròn ở điểm lệch trục?

Hãy tưởng tượng các đường loss là các đường đồng mức quanh nghiệm OLS không regularization. Ta muốn tìm đường loss thấp nhất nhưng vẫn nằm trong miền ràng buộc. Với L2, miền ràng buộc `w1^2 + w2^2 <= t` là hình tròn, biên của nó trơn ở mọi nơi. Điểm tối ưu thường xảy ra khi một đường đồng mức loss tiếp xúc với biên hình tròn. Vì biên tròn không có góc đặc biệt ở trục `w1=0` hay `w2=0`, xác suất điểm tiếp xúc rơi đúng lên trục là rất nhỏ; nó chỉ xảy ra nếu hình học của loss đối xứng/đặc biệt đúng theo trục đó. Ngược lại, L1 có hình thoi với góc nhọn nằm trên trục. Rất nhiều đường đồng mức khi phình ra sẽ chạm góc trước khi chạm cạnh, nên nghiệm dễ rơi vào `w1=0` hoặc `w2=0`. Nói ngắn gọn: L2 có biên trơn nên không ưu tiên trục; L1 có góc nằm trên trục nên hút nghiệm về trục, tạo weight bằng 0.

# Lesson 4-5. Clustering

## L4.1. Clustering là gì?

Clustering là bài toán unsupervised learning.

Input:

```text
D = {x_1, x_2, ..., x_M}
```

Không có label `y`.

Output:

```text
clusters = {C_1, C_2, ..., C_K}
```

Mục tiêu:

```text
Các điểm trong cùng cluster giống nhau.
Các điểm ở cluster khác nhau thì khác nhau.
```

Hai khái niệm:

```text
intra-cluster distance  -> khoảng cách trong cùng cụm, nên nhỏ
inter-cluster distance  -> khoảng cách giữa cụm, nên lớn
```

## L4.2. Distance measures

### Euclidean distance - L2

```text
d(x, z) = sqrt(Σ_j (x_j - z_j)^2)
```

Dùng khi feature numeric và scale đã ổn.

### Manhattan distance - L1

```text
d(x, z) = Σ_j |x_j - z_j|
```

Ít nhạy hơn L2 với một số outlier theo từng chiều.

### Hamming distance

Dùng cho categorical/binary vectors:

```text
d(x, z) = số vị trí mà x_j != z_j
```

Ví dụ:

```text
x = [0, 1, 1, 0]
z = [0, 0, 1, 1]
d_hamming = 2
```

## L4.3. Evaluation của clustering

Vì không có label, đánh giá clustering khó hơn supervised learning.

### SSE

Sum of squared error:

```text
SSE = Σ_i Σ_{x ∈ C_i} ||x - μ_i||^2
```

Trong đó:

- `C_i` là cluster thứ `i`.
- `μ_i` là centroid của cluster `C_i`.

SSE nhỏ nghĩa là các điểm gần centroid hơn.

### Silhouette

Với một điểm `o_i`:

```text
a(i) = khoảng cách trung bình từ o_i tới các điểm cùng cluster
b(i) = khoảng cách trung bình nhỏ nhất từ o_i tới một cluster khác
```

Silhouette:

```text
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Ý nghĩa:

```text
s(i) gần 1   -> điểm được cluster tốt
s(i) gần 0   -> điểm nằm giữa các cluster
s(i) âm      -> điểm có thể bị gán sai cluster
```

## L4.4. K-means

K-means là partition-based clustering.

Input:

```text
D = {x_1, ..., x_M}
K = số cluster muốn chia
```

Output:

```text
K clusters và K centroids
```

Objective:

```text
minimize Σ_{i=1..K} Σ_{x ∈ C_i} ||x - μ_i||^2
```

Trong đó:

```text
μ_i = centroid của cluster C_i
```

### K-means algorithm

```text
Input:
  D, K, distance function

Step 1. Initialization
  Chọn ngẫu nhiên K centroids:
  μ_1, μ_2, ..., μ_K

Step 2. Assignment
  Với mỗi điểm x:
    gán x vào cluster có centroid gần nhất

  c(x) = argmin_j dist(x, μ_j)

Step 3. Update centroid
  Với mỗi cluster C_j:
    μ_j = average của các điểm trong C_j

  μ_j = (1 / |C_j|) Σ_{x ∈ C_j} x

Step 4. Repeat
  Lặp Step 2 và Step 3 đến khi:
    - centroid không đổi nhiều
    - rất ít điểm đổi cluster
    - SSE không giảm đáng kể
```

### Minh họa

```text
Data points:
  x1, x2, x3, x4, x5, x6

K = 2

Initialize:
  μ1 = x1
  μ2 = x5

Assignment:
  C1 = {x1, x2, x3}
  C2 = {x4, x5, x6}

Update:
  μ1 = average(x1, x2, x3)
  μ2 = average(x4, x5, x6)

Repeat until stable.
```

## L4.5. Problems of K-means

### Initial centroids

K-means phụ thuộc mạnh vào khởi tạo.

```text
khởi tạo tốt -> cluster tốt
khởi tạo xấu -> local optimum kém
```

Giải pháp:

```text
Run K-means nhiều lần với random seeds khác nhau.
Chọn kết quả có SSE nhỏ nhất.
```

### Outliers

Centroid là mean, nên outlier có thể kéo centroid đi xa.

Giải pháp:

```text
outlier removal
random sampling
robust distance/algorithm khác
```

### Chọn K

K-means cần nhập trước `K`.

Hai cách chọn K trong slide:

```text
Elbow method
Average silhouette method
```

Elbow:

```text
Chạy K-means với nhiều K.
Tính WSS/SSE cho mỗi K.
Chọn điểm "khuỷu tay" nơi SSE giảm chậm lại.
```

Silhouette:

```text
Chọn K có average silhouette cao.
```

## L4.6. K-means++

K-means++ cải thiện bước chọn centroid ban đầu.

Trực giác:

```text
Centroid đầu chọn random.
Centroid sau ưu tiên chọn điểm ở xa các centroid đã chọn.
```

Thuật toán:

```text
1. Chọn centroid đầu tiên ngẫu nhiên từ D.
2. Với mỗi điểm x, tính D(x)^2:
     D(x) = khoảng cách từ x tới centroid gần nhất đã chọn.
3. Chọn centroid tiếp theo với xác suất tỷ lệ D(x)^2.
4. Lặp đến khi có K centroids.
5. Chạy K-means thường.
```

Vì các centroid ban đầu trải rộng hơn, K-means++ thường hội tụ tốt hơn random initialization.

## L4.7. Hierarchical clustering

Hierarchical clustering tạo cấu trúc phân cấp cụm.

Hai hướng:

```text
Agglomerative:
  bottom-up
  bắt đầu mỗi điểm là một cluster
  rồi merge dần

Divisive:
  top-down
  bắt đầu tất cả trong một cluster
  rồi split dần
```

Agglomerative algorithm:

```text
1. Mỗi điểm là một cluster riêng.
2. Tính khoảng cách giữa các cluster.
3. Merge hai cluster gần nhất.
4. Cập nhật khoảng cách.
5. Lặp đến khi còn một cluster hoặc đạt số cluster mong muốn.
```

Linkage:

```text
single linkage:
  dist(A,B) = min distance giữa điểm thuộc A và điểm thuộc B

complete linkage:
  dist(A,B) = max distance giữa điểm thuộc A và điểm thuộc B

average linkage:
  dist(A,B) = average distance giữa mọi cặp điểm A-B
```

Output thường là dendrogram.

## L4.8. Density-based clustering

Slide nhắc đến DBSCAN như ví dụ density-based method.

Trực giác:

```text
Cluster là vùng có mật độ điểm cao.
Noise/outlier là điểm nằm ở vùng mật độ thấp.
```

DBSCAN dùng:

```text
eps    = bán kính lân cận
minPts = số điểm tối thiểu để coi là vùng dense
```

DBSCAN mạnh khi cluster có hình dạng không cầu, nhưng nhạy với `eps`.

## L4 Q&A log

Chưa có câu hỏi riêng sau khi tạo file.

### Q1. Trong K-means++, `x'` là gì? "Chọn bằng xác suất" nghĩa là sao?

Trong K-means++, `x'` thường chỉ một data point được chọn từ dataset để làm centroid mới. Dấu `'` không phải đạo hàm; nó chỉ là ký hiệu "một điểm x được chọn". K-means++ không nhất thiết chọn điểm xa nhất một cách cứng. Nó chọn ngẫu nhiên nhưng có trọng số: với mỗi điểm `x`, tính `D(x)^2`, trong đó `D(x)` là khoảng cách từ `x` tới centroid gần nhất đã được chọn. Sau đó xác suất chọn điểm `x` làm centroid tiếp theo là `P(x) = D(x)^2 / Σ_z D(z)^2`. Điểm càng xa các centroid hiện có thì `D(x)^2` càng lớn, nên xác suất được chọn càng cao; nhưng vẫn có yếu tố random để tránh quá cứng và giúp khởi tạo đa dạng.

### Q2. Sau khi tính xác suất trong K-means++, chọn điểm cụ thể như thế nào?

Sau khi có `P(x)` cho từng điểm, ta bốc thăm theo phân phối xác suất đó, thường gọi là roulette-wheel sampling hoặc categorical sampling. Cách làm thủ công: sắp các điểm theo thứ tự, tính xác suất cộng dồn, sinh một số ngẫu nhiên `u` trong `[0,1]`, rồi chọn điểm đầu tiên có xác suất cộng dồn lớn hơn hoặc bằng `u`. Ví dụ `P(x1)=0.1`, `P(x2)=0.3`, `P(x3)=0.6`; cộng dồn là `x1:0.1`, `x2:0.4`, `x3:1.0`. Nếu `u=0.35` thì chọn `x2`; nếu `u=0.8` thì chọn `x3`. Trong code, thường dùng hàm như `np.random.choice(points, p=probabilities)`.

### Q3. Nếu K-means++ vẫn random thì khác gì K-means thường?

Khác ở phân phối random. K-means thường chọn centroid ban đầu random đều: mọi điểm có xác suất gần như bằng nhau, nên có thể chọn nhiều centroid nằm sát nhau và bỏ trống vùng dữ liệu khác. K-means++ chọn centroid đầu random, nhưng các centroid sau random có trọng số theo `D(x)^2`, nên điểm càng xa các centroid đã chọn càng có xác suất cao. Vì vậy nó vẫn random, nhưng là random có định hướng để các centroid ban đầu trải rộng hơn. Không phải đảm bảo luôn tối ưu, nhưng thường cho khởi tạo tốt hơn, SSE thấp hơn và hội tụ ổn định hơn K-means random thường.

# Lesson 6. Decision Tree and Random Forest

## L6.1. Decision tree là gì?

Decision tree biểu diễn hàm dự đoán bằng cây.

```text
internal node -> test một feature/attribute
branch        -> một giá trị/kết quả test
leaf          -> class label
```

Ví dụ:

```text
Outlook?
  Sunny:
    Humidity?
      High   -> No
      Normal -> Yes
  Overcast -> Yes
  Rain:
    Wind?
      Strong -> No
      Weak   -> Yes
```

Một path từ root tới leaf tương ứng với rule:

```text
IF Outlook=Sunny AND Humidity=High THEN PlayTennis=No
```

Toàn bộ tree là OR của nhiều rule path.

## L6.2. ID3 algorithm

ID3 là thuật toán greedy, top-down.

Ý tưởng:

```text
Ở mỗi node, chọn feature giúp phân loại dữ liệu trong node tốt nhất.
```

Pseudocode:

```text
ID3(S, Attributes):
  Tạo node hiện tại

  Nếu mọi sample trong S cùng class c:
      return leaf(c)

  Nếu Attributes rỗng:
      return leaf(majority_class(S))

  Chọn attribute A tốt nhất theo Information Gain

  Với mỗi value v của A:
      S_v = {x ∈ S | x_A = v}

      Nếu S_v rỗng:
          gắn leaf majority_class(S)
      Ngược lại:
          gắn subtree ID3(S_v, Attributes \ {A})

  return node(A)
```

ID3 không backtrack. Khi đã chọn attribute ở một node, nó không quay lại sửa quyết định đó.

## L6.3. Entropy

Entropy đo độ hỗn loạn/không thuần nhất của label trong một set `S`.

Với `c` classes:

```text
Entropy(S) = - Σ_{i=1..c} p_i log2(p_i)
```

Trong đó:

```text
p_i = tỷ lệ sample thuộc class i trong S
```

Quy ước:

```text
0 * log2(0) = 0
```

Binary case:

```text
Entropy(S) = -p log2(p) - (1-p) log2(1-p)
```

Ý nghĩa:

```text
Entropy = 0:
  S pure, tất cả sample cùng class.

Entropy cao:
  label lẫn lộn.

Binary entropy max = 1:
  hai class cân bằng 50/50.
```

Ví dụ trong tennis dataset:

```text
S có 14 samples:
  Yes = 9
  No  = 5

Entropy(S)
= -(9/14)log2(9/14) - (5/14)log2(5/14)
≈ 0.94
```

## L6.4. Information Gain

Information Gain đo entropy giảm bao nhiêu nếu split theo attribute `A`.

```text
Gain(S, A)
= Entropy(S) - Σ_{v ∈ Values(A)} (|S_v| / |S|) Entropy(S_v)
```

Trong đó:

```text
S_v = {x ∈ S | x_A = v}
```

Trực giác:

```text
Entropy(S) trước split cao.
Sau split, nếu các subset S_v pure hơn, entropy giảm.
Gain lớn -> split tốt.
```

Ví dụ `Wind`:

```text
S:
  9 Yes, 5 No
  Entropy(S) ≈ 0.94

S_Weak:
  6 Yes, 2 No
  Entropy(S_Weak) ≈ 0.81

S_Strong:
  3 Yes, 3 No
  Entropy(S_Strong) = 1

Gain(S, Wind)
= 0.94 - (8/14)*0.81 - (6/14)*1
≈ 0.048
```

Trong ví dụ slide:

```text
Gain(S, Outlook)     ≈ 0.246
Gain(S, Temperature) ≈ 0.029
Gain(S, Humidity)    ≈ 0.151
Gain(S, Wind)        ≈ 0.048
```

Nên root chọn `Outlook`.

## L6.5. Bias của Information Gain và Gain Ratio

Information Gain ưu tiên attribute có nhiều unique values.

Ví dụ:

```text
StudentID có unique value cho từng sample
```

Split theo `StudentID` sẽ tạo nhiều subset rất nhỏ, thường pure, nên gain cao nhưng không generalize.

Gain Ratio giảm bias này:

```text
GainRatio(S, A) = Gain(S, A) / SplitInformation(S, A)
```

Trong đó:

```text
SplitInformation(S, A)
= - Σ_{v ∈ Values(A)} (|S_v| / |S|) log2(|S_v| / |S|)
```

Nếu attribute tạo quá nhiều nhánh, `SplitInformation` lớn, làm GainRatio giảm.

## L6.6. Overfitting trong decision tree

Cây có thể fit hoàn hảo train data nhưng kém trên test.

Nguyên nhân:

```text
noise/error trong data
cây quá sâu
split theo pattern ngẫu nhiên không có ý nghĩa thật
```

Giải pháp:

```text
early stopping:
  dừng trước khi fit hoàn hảo train set

post-pruning:
  grow full tree rồi prune bằng validation set
```

Post-pruning thường dễ dùng hơn vì khó biết khi nào nên dừng sớm.

## L6.7. Real attributes và missing values

ID3 gốc làm tốt với categorical attributes.

Với real attributes:

```text
Discretization:
  age ∈ [0,100]
  -> [0,18), [18,60), [60,100]
```

Với missing values:

```text
Solution 1:
  fill bằng giá trị phổ biến nhất của attribute.

Solution 2:
  fill bằng giá trị phổ biến nhất trong cùng class.
```

## L6.8. Random Forest

Random Forest là ensemble của nhiều decision trees.

Ý tưởng:

```text
Một cây có thể overfit.
Nhiều cây random khác nhau, vote/average lại, thường ổn định hơn.
```

Ba thành phần chính:

```text
Bagging:
  mỗi tree train trên bootstrap sample từ data gốc.

Feature randomization:
  ở mỗi node, chỉ xét random subset của features.

No pruning:
  mỗi tree thường grow sâu/lớn.
```

### Bootstrap sampling

Với dataset `D` có `M` samples:

```text
D_b = sample M lần từ D với replacement
```

Một sample có thể xuất hiện nhiều lần, một số sample không xuất hiện.

### RF learning algorithm

```text
Input:
  D: training data
  K: number of trees

For t = 1..K:
  1. Tạo bootstrap sample D_t từ D
  2. Train decision tree T_t trên D_t
  3. Ở mỗi node:
       chọn random subset features
       tìm split tốt nhất trong subset đó
  4. Grow tree tới size lớn, không prune

Output:
  forest = {T_1, ..., T_K}
```

### RF prediction

Classification:

```text
ŷ = majority_vote(T_1(x), ..., T_K(x))
```

Probability:

```text
P(class=c | x) = average_t P_t(class=c | x)
```

Regression:

```text
ŷ = (1/K) Σ_t T_t(x)
```

## L6.9. Vì sao Random Forest giảm overfitting?

Một tree có high variance:

```text
đổi data một chút -> cây có thể khác nhiều
```

Random Forest giảm variance bằng averaging:

```text
variance của average nhiều model ít tương quan sẽ thấp hơn variance của một model.
```

Feature randomization làm các tree bớt giống nhau.

Bagging làm mỗi tree thấy data hơi khác.

## L6 Q&A log

Chưa có câu hỏi riêng sau khi tạo file.

# Lesson 7. Neural Networks

## L7.1. Artificial neuron

Một neuron nhận input:

```text
x = [x_1, x_2, ..., x_m]
```

Mỗi input có weight:

```text
w = [w_1, w_2, ..., w_m]
```

Thêm bias:

```text
x_0 = 1
w_0 = bias
```

Net input:

```text
Net = w_0 + w_1 x_1 + ... + w_m x_m
    = Σ_{j=0..m} w_j x_j
```

Output:

```text
Out = f(Net)
```

Trong đó `f` là activation function.

## L7.2. Activation functions

### Hard limiter

```text
Out = 1 nếu Net >= θ
Out = 0 otherwise
```

Hoặc bipolar:

```text
Out = sign(Net)
```

Nhược điểm: không trơn, không dùng tốt cho gradient descent/backprop.

### Sigmoid

```text
σ(x) = 1 / (1 + exp(-x))
```

Đạo hàm:

```text
σ'(x) = σ(x)(1 - σ(x))
```

Output nằm trong `[0,1]`.

### Tanh

```text
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

Đạo hàm:

```text
d tanh(x) / dx = 1 - tanh^2(x)
```

Output nằm trong `[-1,1]`.

### ReLU

```text
ReLU(x) = max(0, x)
```

Ưu điểm:

- Dễ tính.
- Phổ biến trong neural networks.
- Giảm saturation so với sigmoid/tanh ở vùng dương.

## L7.3. Architecture

Một neural network có:

```text
input layer
hidden layer(s)
output layer
```

Fully connected:

```text
mọi output neuron layer trước nối tới mọi neuron layer sau.
```

Feed-forward:

```text
signal đi từ input -> hidden -> output
không có loop quay ngược.
```

Recurrent/feedback:

```text
có kết nối quay lại layer trước hoặc chính nó.
```

Trong project, MLP là feed-forward neural network.

## L7.4. Training objective

Với fixed architecture, training là học weights.

Empirical error:

```text
E(w) = (1/|D|) Σ_{(x,d) ∈ D} loss(d, out(x; w))
```

Trong slide với squared error và `n` output neurons:

```text
E_x(w) = (1/2) Σ_{i=1..n} (d_i - Out_i)^2
```

Với toàn dataset:

```text
E_D(w) = (1/|D|) Σ_{x ∈ D} E_x(w)
```

Mục tiêu:

```text
w* = argmin_w E_D(w)
```

## L7.5. Perceptron

Perceptron là neural network đơn giản nhất: một neuron.

Output:

```text
Out = sign(w^T x)
```

Nó tạo decision boundary tuyến tính:

```text
w_0 + w_1 x_1 + w_2 x_2 = 0
```

Algorithm:

```text
Input:
  D = {(x, d)}, d ∈ {-1, 1}
  η = learning rate

Initialize:
  w small random

Repeat:
  Δw = 0
  For each (x, d) in D:
      Out = sign(w^T x)
      If Out != d:
          Δw = Δw + η(d - Out)x
  w = w + Δw

Until all instances classified correctly or stopping criterion.
```

Perceptron chỉ hội tụ nếu data linearly separable và learning rate phù hợp.

## L7.6. Gradient descent

Gradient:

```text
∇E(w) = [∂E/∂w_1, ∂E/∂w_2, ..., ∂E/∂w_N]^T
```

Gradient chỉ hướng tăng nhanh nhất của error.

Muốn giảm error:

```text
w <- w - η ∇E(w)
```

Trong đó:

- `η` là learning rate.
- Nếu `η` quá lớn, training dao động hoặc vượt optimum.
- Nếu `η` quá nhỏ, training rất chậm.

Incremental/SGD:

```text
for each training instance:
    compute gradient on that instance
    update weights immediately
```

Mini-batch:

```text
sample một batch nhỏ
compute gradient trung bình trên batch
update weights
```

## L7.7. Backpropagation

Backpropagation là thuật toán tính gradient cho multi-layer neural network bằng chain rule.

Nó gồm hai pha:

```text
Forward pass:
  input đi qua network để tạo prediction.

Backward pass:
  tính error ở output rồi truyền gradient ngược về các layer trước.
```

### Network 1 hidden layer

Ký hiệu:

```text
x_j      = input j
z_q      = hidden neuron q
y_i      = output neuron i
w_qj     = weight từ input j tới hidden q
w_iq     = weight từ hidden q tới output i
Out_q    = output của hidden neuron q
Out_i    = output của output neuron i
d_i      = desired output i
```

Forward hidden:

```text
Net_q = Σ_j w_qj x_j
Out_q = f(Net_q)
```

Forward output:

```text
Net_i = Σ_q w_iq Out_q
Out_i = f(Net_i)
```

Error:

```text
E = (1/2) Σ_i (d_i - Out_i)^2
```

### Output layer error signal

Ta cần update `w_iq`.

Gradient descent:

```text
Δw_iq = -η ∂E/∂w_iq
```

Dùng chain rule:

```text
∂E/∂w_iq
= ∂E/∂Out_i * ∂Out_i/∂Net_i * ∂Net_i/∂w_iq
```

Ta có:

```text
∂E/∂Out_i = -(d_i - Out_i)
∂Out_i/∂Net_i = f'(Net_i)
∂Net_i/∂w_iq = Out_q
```

Vậy:

```text
∂E/∂w_iq = -(d_i - Out_i) f'(Net_i) Out_q
```

Đặt:

```text
δ_i = (d_i - Out_i) f'(Net_i)
```

Thì:

```text
Δw_iq = η δ_i Out_q
```

### Hidden layer error signal

Hidden neuron không có label trực tiếp, nên error của nó đến từ các output neurons phía sau.

```text
δ_q = f'(Net_q) Σ_i δ_i w_iq
```

Update weight từ input tới hidden:

```text
Δw_qj = η δ_q x_j
```

### General delta rule

Với connection từ neuron/input `b` tới neuron `a`:

```text
Δw_ab = η δ_a Out_b
```

Trong đó:

- `δ_a` là error signal của neuron nhận.
- `Out_b` là output của neuron gửi.

## L7.8. Backpropagation algorithm

```text
Input:
  D: training data
  η: learning rate
  network architecture

Initialize:
  weights random small values

Repeat for epochs:
  For each training sample (x, d):

    Forward:
      compute all Net and Out values layer by layer

    Output error:
      δ_output = (d - Out) * f'(Net)

    Backward:
      for hidden layers from last to first:
          δ_hidden = f'(Net_hidden) * weighted_sum(δ_next)

    Update:
      for every connection b -> a:
          w_ab = w_ab + η δ_a Out_b

Until:
  error < threshold
  or max epochs
  or validation performance stops improving
```

## L7.9. Momentum

Gradient descent có thể chậm hoặc dao động.

Momentum thêm thành phần update cũ:

```text
Δw(t+1) = -η ∇E(t+1) + α Δw(t)
```

Trong đó:

- `α ∈ [0,1]` là momentum parameter.
- Thường `α ≈ 0.9`.

Trực giác:

```text
Nếu nhiều bước gradient đi cùng hướng,
momentum tăng tốc.

Nếu gradient dao động zig-zag,
momentum làm quỹ đạo mượt hơn.
```

## L7.10. Practical issues

### Weight initialization

Weights thường khởi tạo random nhỏ.

Nếu quá lớn:

```text
sigmoid/tanh dễ saturation
gradient nhỏ
training mắc kẹt
```

### Learning rate

```text
η lớn  -> nhanh nhưng dễ vượt optimum / dao động
η nhỏ  -> ổn định nhưng chậm
```

### Number of hidden neurons

Không có rule chung tuyệt đối.

```text
quá ít neuron  -> underfit
quá nhiều      -> overfit, train chậm, khó giải thích
```

### Black-box issue

ANN thường chính xác cao nhưng khó giải thích nội bộ.

## L7 Q&A log

Chưa có câu hỏi riêng sau khi tạo file.

# Lesson 8. Support Vector Machines

## L8.1. SVM là gì?

SVM ban đầu là phương pháp classification tuyến tính hai lớp.

Training data:

```text
D = {(x_1, y_1), ..., (x_r, y_r)}
x_i ∈ R^n
y_i ∈ {-1, +1}
```

SVM tìm hyperplane:

```text
f(x) = <w, x> + b = 0
```

Decision:

```text
predict +1 nếu <w, x> + b >= 0
predict -1 otherwise
```

## L8.2. Hyperplane và margin

Hyperplane:

```text
H0: <w, x> + b = 0
```

Hai marginal hyperplanes:

```text
H+: <w, x> + b = +1
H-: <w, x> + b = -1
```

Với linearly separable data:

```text
y_i(<w, x_i> + b) >= 1
```

Khoảng cách từ điểm `x` tới hyperplane `H0`:

```text
distance(x, H0) = |<w, x> + b| / ||w||
```

Khoảng cách từ `H+` tới `H0`:

```text
d+ = 1 / ||w||
```

Khoảng cách từ `H-` tới `H0`:

```text
d- = 1 / ||w||
```

Margin:

```text
margin = d+ + d- = 2 / ||w||
```

Muốn margin lớn nhất nghĩa là muốn `||w||` nhỏ nhất.

## L8.3. Hard-margin SVM primal problem

Maximize margin:

```text
maximize 2 / ||w||
```

Tương đương:

```text
minimize (1/2)||w||^2
```

Subject to:

```text
y_i(<w, x_i> + b) >= 1
for all i
```

Đây là constrained convex optimization.

## L8.4. Support vectors

Support vectors là các điểm nằm trên hoặc sát margin, có ảnh hưởng trực tiếp đến hyperplane.

Trong hard-margin:

```text
y_i(<w, x_i> + b) = 1
```

Các điểm xa margin thường không ảnh hưởng tới nghiệm cuối.

Trực giác:

```text
SVM không quan tâm mọi điểm như nhau.
Nó quan tâm các điểm khó nhất, nằm gần boundary nhất.
```

## L8.5. Lagrange multipliers

Để giải bài toán có ràng buộc, dùng Lagrange multipliers.

Hard-margin primal:

```text
minimize  (1/2)||w||^2
subject to y_i(<w,x_i> + b) - 1 >= 0
```

Lagrangian:

```text
L(w,b,α)
= (1/2)||w||^2
  - Σ_i α_i [y_i(<w,x_i> + b) - 1]
```

Trong đó:

```text
α_i >= 0
```

KKT cho biết tại optimum:

```text
α_i [y_i(<w,x_i> + b) - 1] = 0
```

Nghĩa là:

```text
nếu điểm không nằm trên margin:
  y_i(<w,x_i> + b) > 1
  -> α_i = 0

nếu α_i > 0:
  điểm đó là support vector
```

## L8.6. Dual form

Từ Lagrangian, lấy đạo hàm theo `w` và `b`, set bằng 0:

```text
∂L/∂w = 0
=> w = Σ_i α_i y_i x_i

∂L/∂b = 0
=> Σ_i α_i y_i = 0
```

Thay vào Lagrangian, được dual problem:

```text
maximize
  Σ_i α_i - (1/2) Σ_i Σ_j α_i α_j y_i y_j <x_i, x_j>

subject to
  α_i >= 0
  Σ_i α_i y_i = 0
```

Sau khi giải `α`, classifier:

```text
f(z) = Σ_i α_i y_i <x_i, z> + b
```

Chỉ các điểm có `α_i > 0` xuất hiện trong classifier.

## L8.7. Soft-margin SVM

Thực tế data thường không linearly separable.

Ta thêm slack variables:

```text
ξ_i >= 0
```

Constraint:

```text
y_i(<w,x_i> + b) >= 1 - ξ_i
```

Ý nghĩa:

```text
ξ_i = 0:
  điểm nằm đúng ngoài margin an toàn

0 < ξ_i <= 1:
  điểm nằm trong margin nhưng vẫn phân loại đúng

ξ_i > 1:
  điểm bị phân loại sai
```

Soft-margin objective:

```text
minimize (1/2)||w||^2 + C Σ_i ξ_i

subject to:
  y_i(<w,x_i> + b) >= 1 - ξ_i
  ξ_i >= 0
```

`C` điều khiển trade-off:

```text
C nhỏ:
  ưu tiên margin lớn
  cho phép nhiều lỗi hơn

C lớn:
  phạt lỗi mạnh
  cố fit training data hơn
  dễ overfit nếu data noisy
```

Soft-margin cũng tương đương tối ưu hinge loss:

```text
minimize
  (1/2)||w||^2 + C Σ_i max(0, 1 - y_i(<w,x_i> + b))
```

Hinge loss:

```text
loss_i = max(0, 1 - y_i score_i)
score_i = <w,x_i> + b
```

Nếu:

```text
y_i score_i >= 1
```

thì loss = 0.

## L8.8. Kernel trick

Nếu data không linearly separable trong input space, ta có thể map sang feature space:

```text
φ: x -> φ(x)
```

Rồi chạy linear SVM trong feature space.

Classifier:

```text
f(z) = Σ_i α_i y_i <φ(x_i), φ(z)> + b
```

Vấn đề:

```text
φ(x) có thể rất cao chiều
tính trực tiếp rất tốn
```

Kernel trick:

```text
K(x,z) = <φ(x), φ(z)>
```

Ta không cần tính `φ(x)` trực tiếp, chỉ cần tính kernel.

Classifier:

```text
f(z) = Σ_i α_i y_i K(x_i, z) + b
```

Ví dụ kernel:

```text
linear:
  K(x,z) = x^T z

polynomial:
  K(x,z) = (x^T z + c)^d

RBF/Gaussian:
  K(x,z) = exp(-γ ||x-z||^2)
```

## L8.9. Multiclass SVM

SVM gốc là binary classifier.

Với nhiều classes, thường dùng:

```text
one-vs-rest:
  train K classifiers, mỗi classifier phân biệt class k với phần còn lại

one-vs-one:
  train classifier cho từng cặp class
```

## L8.10. SVM summary

```text
SVM tìm hyperplane có margin lớn nhất.
Hard-margin dùng khi data linearly separable.
Soft-margin thêm slack để chịu noise/overlap.
C điều khiển trade-off giữa margin và lỗi train.
Dual form làm classifier phụ thuộc vào support vectors.
Kernel trick cho phép học nonlinear boundary.
```

So sánh nhanh với Ridge:

```text
Ridge λ lớn:
  phạt weight lớn mạnh hơn, model đơn giản hơn.

SVM C lớn:
  phạt lỗi mạnh hơn, cố fit train hơn.
```

Vì vậy:

```text
λ trong Ridge càng lớn -> regularization càng mạnh.
C trong SVM càng lớn -> regularization càng yếu hơn theo nghĩa cho phép ít lỗi hơn.
```

## L8 Q&A log

Chưa có câu hỏi riêng sau khi tạo file.

# Cross-Lesson Map

## Model nào thuộc lesson nào?

```text
Linear Regression / Ridge / LASSO -> L3
K-means / Hierarchical / DBSCAN   -> L4-5
Decision Tree / Random Forest     -> L6
MLP / Neural Network              -> L7
SVM                               -> L8
```

## Công thức loss quan trọng

```text
Linear regression square loss:
  (y - f(x))^2

Ridge:
  ||y - Aw||^2 + λ||w||_2^2

LASSO:
  ||y - Aw||^2 + λ||w||_1

Neural network squared error:
  (1/2)Σ_i(d_i - Out_i)^2

SVM hinge loss:
  max(0, 1 - y_i(<w,x_i> + b))
```

## Thuật toán train theo kiểu nào?

```text
OLS:
  nghiệm đóng bằng normal equation

Ridge:
  nghiệm đóng với (A^T A + λI)^(-1)

LASSO:
  tối ưu số, thường không có nghiệm đóng đơn giản

K-means:
  lặp assignment/update centroid

ID3:
  greedy split bằng information gain

Random Forest:
  bagging nhiều cây + random feature subset

Neural Network:
  gradient descent + backpropagation

SVM:
  constrained convex optimization / dual quadratic programming
```

## Liên hệ với project dog eye color

Project hiện dùng hoặc liên quan trực tiếp:

```text
Logistic Regression:
  họ hàng với linear models, nhưng dùng cho classification.

Random Forest:
  L6, ensemble nhiều decision trees.

MLP:
  L7, neural network train bằng gradient descent/backprop.

TabNet:
  gần L7/L6, neural network tabular có attention mask chọn feature.

TabPFN/TabICL:
  foundation models hiện đại, không nằm trong slide cơ bản nhưng dùng nhiều ý tưởng từ L7/L8:
  vector, loss, attention, pretrained weights, inference.
```

## Q&A log chung

### Q1. File này được tạo để làm gì?

File này được tạo để tóm tắt các lesson PDF theo cùng phong cách với `tabicl_knowledge.md`: giải thích chi tiết công thức và thuật toán, có minh họa, và có Q&A log theo từng lesson để sau này thêm câu hỏi đúng chỗ.

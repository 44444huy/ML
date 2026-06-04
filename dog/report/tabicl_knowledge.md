# TabICL Knowledge Notes

Mục tiêu của tài liệu này: gom lại kiến thức về TabICL theo cách người chưa biết project vẫn có thể đọc từ đầu. Mỗi câu hỏi mới về TabICL/project sẽ được tóm tắt ngắn gọn và nối thêm vào phần "Q&A log".

Nguồn chính:
- Official repo: https://github.com/soda-inria/tabicl
- TabICL paper: https://arxiv.org/abs/2502.05564
- TabICL docs: https://tabicl.readthedocs.io/

## 1. Bối cảnh project

Project FDP-EVC dự đoán đặc điểm nhìn thấy được từ DNA.

- Part A `human/`: dùng 41 HIrisPlex-S SNP để dự đoán eye/hair/skin trên người. Nhãn là silver label, tức là nhãn do model HIrisPlex-S sinh ra, không phải ground truth thật.
- Part B `dog/`: dùng dữ liệu chó có phenotype thật. Hiện pipeline chính đã làm là dog eye color: dự đoán mắt xanh/brown từ genotype SNP.

Dog eye dataset hiện tại:

- `X`: ma trận genotype, kích thước khoảng `2769 dogs x 56 SNPs`.
- `y`: nhãn nhị phân, `1 = blue eyes`, `0 = brown/non-blue eyes`.
- Positive rate khoảng `3.9%`, nên đây là bài toán mất cân bằng mạnh.

Các model hiện có trong project:

- Majority baseline: luôn đoán class âm.
- Logistic Regression và Random Forest: baseline cổ điển.
- MLP: neural network đơn giản, train bằng `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`.
- TabPFN: tabular foundation model, dự đoán theo in-context learning.
- TabNet: neural tabular model có attention để chọn feature.

TabICL sẽ được thêm như một tabular foundation model mới để so sánh với các model trên.

## 2. SNP genotype là gì trong project

Mỗi dog có nhiều SNP feature. Với một dog và một SNP, giá trị thường là một trong ba số:

```text
0 = không có allele đang được đếm
1 = có 1 bản sao allele đó
2 = có 2 bản sao allele đó
```

Ví dụ một dòng dữ liệu:

```text
dog_1 = [SNP_1=0, SNP_2=2, SNP_3=1, ..., SNP_56=0]
```

Khi viết:

```text
SNP_1 = [0, 1, 0, 2, 1, ...]
```

thì đây không phải là một SNP của một dog. Đây là cả một cột `SNP_1` nhìn trên nhiều dog khác nhau.

Ví dụ:

| Dog | SNP_1 | SNP_2 | SNP_3 |
|---|---:|---:|---:|
| dog_1 | 0 | 2 | 1 |
| dog_2 | 1 | 2 | 0 |
| dog_3 | 0 | 1 | 2 |
| dog_4 | 2 | 2 | 1 |
| dog_5 | 1 | 0 | 0 |

Nhìn theo row:

```text
dog_1 = [0, 2, 1]
```

Nhìn theo column:

```text
SNP_1 = [0, 1, 0, 2, 1]
```

## 3. TabICL là gì

TabICL là một tabular foundation model. Nó được pretrain trước trên nhiều dataset tổng hợp, rồi khi gặp dataset mới, nó dự đoán bằng in-context learning.

Nói đơn giản:

```text
MLP / TabNet:
    train trên dataset của mình -> cập nhật weights -> predict

TabICL:
    đã pretrain từ trước
    nhận X_train, y_train, X_test làm context
    predict trong một forward pass
    không cần cập nhật weights chính trên dataset của mình
```

Vì vậy, TabICL giống TabPFN ở ý tưởng lớn: model không chỉ học một bài toán cụ thể, mà học một "cách học" cho dữ liệu bảng.

## 4. In-context learning là gì

In-context learning nghĩa là model học quy luật từ các ví dụ được đưa vào context lúc dự đoán.

Ví dụ:

```text
Context:
    dog_1 features -> brown
    dog_2 features -> blue
    dog_3 features -> brown
    ...
    dog_test features -> ?

Output:
    P(blue | dog_test, context)
```

TabICL không dùng gradient để train lại trên dog dataset như MLP. Nó dùng attention để so sánh test sample với các training samples trong context.

## 5. Kiến trúc tổng quan của TabICL

TabICL xử lý bảng qua ba bước lớn:

```text
X_train, y_train, X_test
        |
        v
1. Column-wise embedding
        |
        v
2. Row-wise interaction
        |
        v
3. Dataset-wise in-context learning
        |
        v
Predicted probabilities
```

Giải thích ngắn:

- Column-wise embedding: hiểu từng cột feature dựa trên phân phối của cột đó.
- Row-wise interaction: ghép các feature trong cùng một sample để tạo row embedding.
- Dataset-wise ICL: dùng train rows có label để dự đoán test rows.

## 6. Set Transformer là gì

Set Transformer là một dạng Transformer dùng cho dữ liệu dạng set.

Set nghĩa là tập hợp không có thứ tự. Với một cột SNP:

```text
SNP_A = [0, 0, 0, 1, 0]
```

Nếu đổi thứ tự các dog:

```text
SNP_A = [0, 1, 0, 0, 0]
```

thì phân phối của cột vẫn giống nhau. Thứ tự row không quan trọng. Vì vậy, TabICL có thể xem một cột như một set các giá trị.

Set Transformer dùng attention để mỗi giá trị trong cột "nhìn" các giá trị khác trong cùng cột.

Ví dụ:

```text
SNP_A = [0, 0, 0, 1, 0]
```

Giá trị `1` có thể học được:

```text
"Trong cột này, 1 là giá trị hiếm."
```

Giá trị `0` có thể học được:

```text
"Trong cột này, 0 là giá trị phổ biến."
```

## 7. Distribution-aware column-wise embedding

Đây là ý quan trọng nhất lúc mới học TabICL.

MLP thường nhìn một cell như một số:

```text
SNP_A = 1
SNP_B = 1
```

Cả hai đều là số `1`. MLP vẫn phân biệt được vì chúng nằm ở hai vị trí feature khác nhau, nhưng bản thân giá trị `1` không tự mang thông tin về việc nó hiếm hay phổ biến trong cột.

TabICL nhìn thêm phân phối của cả cột.

Ví dụ:

```text
SNP_A = [0, 0, 0, 1, 0]  -> giá trị 1 rất hiếm
SNP_B = [0, 1, 2, 1, 2]  -> giá trị 1 khá bình thường
```

Cùng là `1`, nhưng ý nghĩa khác nhau:

```text
SNP_A=1 có thể là tín hiệu mạnh hơn SNP_B=1
```

Vì vậy TabICL tạo embedding khác nhau:

```text
embedding(SNP_A=1) != embedding(SNP_B=1)
```

Đây gọi là distribution-aware: biểu diễn một cell dựa trên cả giá trị của nó và phân phối của cột chứa nó.

## 8. So sánh nhanh với model trong project

| Model | Cách học | Điểm chính |
|---|---|---|
| Logistic Regression | Train trực tiếp trên dog dataset | Baseline tuyến tính |
| Random Forest | Train trực tiếp trên dog dataset | Bắt non-linear interaction tốt hơn LR |
| MLP | Train trực tiếp trên dog dataset | Dùng `pos_weight` để xử lý mất cân bằng |
| TabNet | Train trực tiếp trên dog dataset | Attention chọn feature quan trọng |
| TabPFN | Pretrained, in-context learning | Foundation model cho tabular |
| TabICL | Pretrained, in-context learning | Column-wise + row-wise + dataset-wise Transformer |

Với dog eye color, TabICL nên được xem là một model so sánh với TabPFN:

```text
TabPFN vs TabICL:
    cả hai đều là tabular foundation model
    cả hai không train lại weights chính trên dataset của mình
    cả hai nên được đánh giá bằng PR-AUC, ROC-AUC, F1
    vì dữ liệu mất cân bằng, nên cân nhắc threshold tuning trên validation
```

## 9. Thuật toán TabICL ở mức dễ hiểu

```text
Input:
    X_train, y_train, X_test

Step 1: Column-wise embedding
    Với mỗi cột SNP:
        xem toàn bộ giá trị trong cột như một set
        dùng Set Transformer để tạo embedding có hiểu phân phối cột

Step 2: Row-wise interaction
    Với mỗi dog:
        gom embedding của các SNP trong dog đó
        dùng Transformer để học tương tác giữa SNPs
        tạo row embedding cho dog

Step 3: Dataset-wise ICL
    Gắn label embedding cho train rows
    Test rows có label bị mask/unknown
    Transformer cho test rows attention tới train rows

Step 4: Prediction
    Xuất xác suất:
        P(blue)
        P(brown/non-blue)
```

## 10. Q&A log

### Q1. Vì sao viết `SNP_1 = [0, 1, 0, 2, 1, ...]` trong khi mỗi SNP chỉ là 0/1/2?

Vì đó là nhìn theo cột trên nhiều sample. Với một dog, mỗi SNP chỉ là một số `0/1/2`. Nhưng với cả dataset, một cột SNP là danh sách giá trị của SNP đó trên toàn bộ các dog.

### Q2. Set Transformer là gì và vì sao TabICL dùng nó cho từng cột?

Set Transformer là Transformer cho dữ liệu dạng tập hợp không có thứ tự. Một cột SNP có thể được xem như set các giá trị vì thứ tự row không quan trọng. TabICL dùng Set Transformer để hiểu phân phối của cột, ví dụ biết `1` trong một cột là hiếm hay phổ biến.

### Q3. Distribution-aware column-wise embedding nghĩa là gì?

Nghĩa là embedding của một cell phụ thuộc vào cả giá trị cell và phân phối của cột. Cùng là `1`, nhưng `1` trong cột hiếm khác với `1` trong cột phổ biến. TabICL dùng thông tin này trước khi ghép các feature thành row embedding.

### Q4. Tài liệu này sẽ được cập nhật như thế nào?

Từ thời điểm tạo file này, mỗi câu hỏi mới về TabICL hoặc cách đưa TabICL vào project sẽ được trả lời trong chat, rồi tóm tắt ngắn gọn vào phần Q&A log. Mục tiêu là biến file này thành tài liệu học dần: người mới đọc từ đầu có thể hiểu dữ liệu, project, model, thuật toán và lý do chọn cách triển khai.

### Q5. Embedding thành vector là gì? Column-wise embedding là gì?

Embedding là cách đổi một giá trị thô thành một vector số giàu thông tin hơn. Ví dụ `SNP_A=1` ban đầu chỉ là một số, nhưng sau embedding có thể thành một vector như `[0.2, -0.7, 1.1, ...]`. Vector này không phải genotype mới; nó là biểu diễn mà model dùng để tính toán. Column-wise embedding nghĩa là TabICL tạo embedding cho cell bằng cách nhìn cả cột chứa cell đó, nên `1` trong cột hiếm và `1` trong cột phổ biến có thể nhận hai vector khác nhau.

### Q6. Attention là gì?

Attention là cơ chế để model tự tính "nên chú ý vào đâu nhiều hơn". Với một test dog, attention có thể gán trọng số cao cho các train dog có SNP pattern giống nó và trọng số thấp cho các train dog ít liên quan. Công thức trực giác là: so sánh query của test sample với key của từng train sample, biến điểm giống nhau thành trọng số, rồi lấy tổng có trọng số các value. Trong TabICL, attention giúp test rows học từ các training rows trong context mà không cần train lại weights bằng gradient.

### Q7. Attention của TabICL khác gì TabNet? Vì sao phải lấy tổng có trọng số thay vì lấy label của dog giống nhất?

TabNet dùng attention chủ yếu để chọn feature/SNP quan trọng trong từng decision step, ví dụ bước này chú ý SNP chr18 nhiều hơn SNP khác. Sau khi TabNet train xong, lúc predict nó không so dog test với từng dog train nữa. TabICL thì đưa cả train rows có label và test rows vào context, rồi dùng attention để test row học từ các train rows liên quan. Bước lấy tổng có trọng số không phải lấy trung bình nhãn thô đơn giản; đó là cách Transformer gom thông tin từ nhiều sample liên quan. Nó giống soft k-nearest-neighbor hơn là 1-nearest-neighbor: nhiều dog hơi giống nhau cùng vote sẽ ổn định hơn một dog giống nhất nhưng có thể nhiễu, và kết quả cũng cho xác suất mềm thay vì hard label.

### Q8. Attention weight lấy từ đâu?

Attention weight không phải số nhập tay. Model tự tính từ embedding bằng các ma trận đã học trong pretraining. Trực giác: test row được biến thành query, mỗi train row được biến thành key. Model tính điểm giống nhau giữa query và từng key, rồi dùng softmax để biến các điểm này thành trọng số dương có tổng bằng 1. Trong ví dụ `0.45, 0.30, 0.15, 0.10`, đó chỉ là số minh họa cho các attention weights mà model có thể tự tạo ra.

### Q9. Embedding và similarity giữa hai vector hoạt động sâu hơn như thế nào?

Embedding biến dữ liệu thô thành vector trong một không gian latent, nơi các chiều không nhất thiết có nghĩa thủ công như "SNP hiếm" hay "SNP chr18", nhưng được model học để hữu ích cho dự đoán. Với attention, row embedding của test sample được chiếu thành query `q`, row embedding của train sample được chiếu thành key `k`, rồi similarity thường được tính bằng scaled dot product: `score = q . k / sqrt(d)`. Dot product lớn khi hai vector cùng hướng và có magnitude phù hợp. Softmax biến các score thành attention weights. Cosine similarity là phiên bản chỉ nhìn hướng vector, còn attention thường dùng dot product đã học sau các phép chiếu `W_Q`, `W_K`, `W_V`.

### Q10. Vector embedding của một SNP cell được tạo ra như thế nào?

Vector không có sẵn trong SNP. Nó được tạo bởi các weight đã học của model. Cách đơn giản để hiểu là một bảng tra embedding: `0`, `1`, `2` mỗi giá trị được map sang một vector trainable khác nhau. Trong model thật, nhất là TabICL, bước này phức tạp hơn: scalar value được đưa qua các phép biến đổi neural, rồi Set Transformer xử lý cả cột để output embedding của từng cell. Vì vậy embedding của `SNP_A=1` không chỉ phụ thuộc vào số `1`, mà còn phụ thuộc vào các giá trị khác trong cột `SNP_A`. Các weight sinh embedding đã được học trong pretraining bằng mục tiêu dự đoán label trên nhiều bảng dữ liệu synthetic.

### Q11. Row-wise interaction, [CLS] token, positional encoding và feature permutation ensemble là gì?

Sau column-wise embedding, mỗi dog không còn là 56 số `0/1/2`, mà là 56 vector cell embeddings. Row-wise interaction dùng Transformer theo chiều feature để các SNP trong cùng một dog trao đổi thông tin với nhau, ví dụ SNP vùng ALX4 có thể kết hợp với SNP khác. TabICL thêm bốn token học được gọi là `[CLS]` vào mỗi row; các token này attention tới các SNP tokens và gom thông tin thành một row embedding duy nhất cho dog. Positional encoding, cụ thể RoPE, thêm thông tin vị trí feature để tránh collapse khi nhiều SNP có phân phối giống nhau. Vì thứ tự cột trong tabular data vốn không có nghĩa tự nhiên, TabICL giảm phụ thuộc vào thứ tự bằng cách chạy nhiều hoán vị cột và trung bình dự đoán.

### Q12. Trong cùng một cột, các cell cùng giá trị như `0` có embedding giống nhau không?

Về trực giác, các cell cùng giá trị trong cùng một cột nên có embedding rất giống nhau vì chúng có cùng scalar value và cùng column distribution. Tuy nhiên trong TabICL, embedding được tạo bởi cơ chế theo cell và có thể phụ thuộc vào vị trí sample trong quá trình xử lý set/context, nên không nên hiểu cứng là mọi `0` chắc chắn có vector y hệt nhau. Điều quan trọng hơn là: tất cả các `0` trong `SNP_A=[0,0,0,1,0]` đều mang ý nghĩa "giá trị phổ biến trong cột SNP_A", còn `1` mang ý nghĩa "giá trị hiếm trong cột SNP_A". Sau bước row-wise interaction, mỗi dog còn có ngữ cảnh các SNP khác của chính dog đó, nên hai dog cùng `SNP_A=0` vẫn có row embedding khác nhau.

### Q13. Phân phối cột là gì?

Phân phối cột là cách các giá trị trong một feature xuất hiện trên toàn dataset. Với SNP, vì giá trị thường là `0/1/2`, phân phối cột có thể hiểu rất cụ thể là số lượng hoặc tỉ lệ của `0`, `1`, `2` trong cột đó. Ví dụ `SNP_A=[0,0,0,1,0]` có phân phối `0:80%`, `1:20%`, `2:0%`; còn `SNP_B=[0,1,2,1,2]` có phân phối `0:20%`, `1:40%`, `2:40%`. TabICL dùng phân phối này để hiểu cùng một giá trị như `1` là hiếm hay phổ biến trong từng cột.

### Q14. Transformer tạo `e1'`, `e2'`, `e3'` từ `e1`, `e2`, `e3` bằng cách nào?

Một Transformer layer tạo embedding mới bằng self-attention và feed-forward network. Với mỗi token như `e1`, model tạo query `q1 = W_Q e1`; với tất cả token, model tạo key `k_j = W_K e_j` và value `v_j = W_V e_j`. Sau đó tính điểm liên quan `score_1j = q1 . k_j / sqrt(d)`, dùng softmax để ra attention weights, rồi gom thông tin `z1 = sum_j alpha_1j v_j`. Cuối cùng dùng residual connection, layer norm và feed-forward network để tạo `e1'`. Làm tương tự cho `e2`, `e3`. Multi-head attention làm nhiều phép attention song song để học nhiều kiểu tương tác SNP khác nhau.

### Q15. Head trong multi-head attention là gì?

Một head là một bộ attention riêng, có ma trận `W_Q`, `W_K`, `W_V` riêng. Nếu một attention head giống một "cách nhìn" vào row SNP, thì multi-head attention là nhiều cách nhìn chạy song song. Cùng một input `[e1,e2,e3]`, head 1 có thể tạo attention weights kiểu này, head 2 kiểu khác, vì chúng dùng parameters khác nhau. Output của các head được nối lại rồi trộn bằng một ma trận học được. Nhờ vậy Transformer không bị giới hạn vào một loại quan hệ duy nhất giữa SNPs.

### Q16. Attention có phải chỉ học tương tác từng cặp SNP không?

Attention score đúng là được tính theo từng cặp token, ví dụ `SNP_1` so với `SNP_2`, `SNP_1` so với `SNP_3`. Nhưng output của một token là tổng có trọng số của nhiều value cùng lúc, nên `e1'` có thể chứa thông tin từ cả `SNP_1`, `SNP_2`, `SNP_3`. Sau nhiều Transformer layers, mỗi token đã chứa thông tin trộn từ nhiều token khác, nên layer sau có thể học tương tác bậc cao hơn như tổ hợp `SNP_1 + SNP_2 + SNP_3`. Các `[CLS]` tokens cũng gom thông tin từ toàn bộ row, nên row embedding cuối cùng là biểu diễn của cả pattern SNP, không chỉ một cặp SNP.

### Q17. Vì sao lại có nhiều Transformer layers?

Transformer thường không chỉ có một layer mà là một stack nhiều layer. Mỗi layer gồm self-attention để trộn thông tin giữa tokens và feed-forward network để biến đổi biểu diễn. Layer đầu học các quan hệ trực tiếp hơn giữa SNPs; layer sau nhận các embeddings đã được trộn thông tin từ layer trước, nên có thể học pattern phức tạp hơn. Vì vậy nhiều layer giống như nhiều vòng refine representation: từ cell embedding thô, đến SNP embedding có ngữ cảnh, đến row embedding giàu thông tin.

### Q18. Có thể hiểu bước 1 và bước 2 là nhìn theo chiều dọc và chiều ngang không?

Có. Đây là cách nhớ rất tốt. Bước 1 column-wise embedding nhìn theo chiều dọc của bảng: với mỗi SNP column, xem phân phối giá trị `0/1/2` qua nhiều dog để tạo cell embeddings có ngữ cảnh cột. Bước 2 row-wise interaction nhìn theo chiều ngang: với mỗi dog row, cho các SNP embeddings trong cùng dog tương tác với nhau để tạo row embedding đại diện cho sample. Sau đó bước 3 dataset-wise ICL nhìn ở mức dataset: test dog học từ các train dogs có label trong context.

### Q19. Positional encoding / "dấu vị trí" nghĩa là gì?

Transformer nếu chỉ nhận một danh sách embeddings có thể không tự biết token nào là feature thứ nhất, token nào là feature thứ hai. Positional encoding là cách thêm tín hiệu vị trí vào embedding, giống dán nhãn "cột 1", "cột 2" cho từng SNP token trước khi attention. Ví dụ nếu `SNP_A=0` và `SNP_B=0` có cell embedding giống nhau, sau khi thêm positional encoding chúng trở thành `embedding(0) + position_1` và `embedding(0) + position_2`, nên model phân biệt được "0 ở SNP_A" khác "0 ở SNP_B". RoPE là một dạng positional encoding dùng phép xoay vector theo vị trí trong attention, thay vì chỉ cộng vector vị trí đơn giản.

### Q20. Bước 3 dataset-wise in-context learning hoạt động như thế nào?

Sau bước 2, mỗi dog đã có một `row_embedding` đại diện cho toàn bộ pattern SNP của dog đó. Bước 3 tạo một context gồm train dogs có label và test dog chưa có label. Train rows được gắn thêm label embedding, ví dụ blue hoặc brown; test rows được gắn label mask/unknown. Dataset-wise Transformer cho test row attention tới các train rows để gom thông tin từ những dog liên quan. Output cuối cùng của test row đi qua prediction head để tạo xác suất class, ví dụ `P(blue)` và `P(brown)`. Đây là in-context learning vì model dùng các ví dụ train trong context để dự đoán test, thay vì cập nhật weights trên dog dataset.

### Q21. Label embedding được tạo thế nào? Bước 3 có giống attention đã học trước đó không?

Label embedding cũng là vector học được bởi model. Với classification, có thể hiểu đơn giản là một bảng tra: `brown -> vector_brown`, `blue -> vector_blue`, và test row dùng một vector đặc biệt như `unknown/mask`. Các vector này là parameters đã học trong pretraining, không phải lấy từ dữ liệu thô. Bước 3 dùng cùng ý tưởng attention đã học trước đó, nhưng đổi cấp độ: row-wise attention là SNP token nhìn SNP token trong cùng một dog; dataset-wise attention là dog/test row token nhìn các train dog tokens trong context. Công thức query-key-value giống nhau, chỉ khác token đại diện cho cái gì.

### Q22. TabPFN và TabICL khác nhau như thế nào?

TabPFN và TabICL đều là tabular foundation models dùng in-context learning: đưa `X_train, y_train, X_test` vào model đã pretrained và dự đoán không cần train lại weights chính trên dataset mới. Khác biệt lớn nằm ở cách mã hóa bảng. TabPFN là Prior-Data Fitted Network, được pretrained trên dữ liệu synthetic từ prior để xấp xỉ Bayesian prediction cho tabular tasks nhỏ. TabICL thiết kế kiến trúc rõ theo cấu trúc bảng: bước 1 nhìn dọc theo cột để tạo distribution-aware cell embeddings, bước 2 nhìn ngang theo row để tạo row embeddings, bước 3 làm dataset-wise ICL giữa train/test rows. Vì vậy TabICL dễ giải thích theo ba chiều của bảng hơn và được thiết kế để scale tốt hơn với dữ liệu lớn, còn TabPFN nổi tiếng vì rất mạnh/nhanh trên small tabular classification.

### Q23. TabICL đã được thêm vào project như thế nào?

TabICL được thêm như một optional foundation-model baseline cho dog eye color. Script mới là `dog/src/train/train_tabicl.py`, đọc cùng `eye_processed.npz` và `eye_splits.json`, chạy 5-fold CV trên trainval, refit trên full trainval, rồi ghi `dog/experiments/eye/tabicl_results.json`. Vì TabICL không có `pos_weight`, script dùng cùng cách với TabPFN: standardize feature bằng train statistics và tune threshold trên validation để tối ưu F1. `dog/src/evaluation/report_eye.py` sẽ tự thêm TabICL vào bảng/figures nếu file kết quả `tabicl_results.json` tồn tại.

### Q24. Có phải chỉ TabICL mới cần chạy `.venv` không? Vì sao diagram lúc đầu chưa đủ?

Không. `.venv` không thuộc riêng TabICL; đó là môi trường Python chung để chạy toàn bộ project với đúng dependency (`tabpfn`, `tabicl`, `pytorch-tabnet`, `torch`, `sklearn`, ...). Các model cũ cũng có thể và nên chạy bằng cùng `.venv` để tránh lệch package giữa máy và project. Lúc đầu bar chart đã đủ vì nó đọc metric từ các file JSON có sẵn, nhưng PR curve phải chạy lại `predict_proba` của từng model để lấy điểm xác suất trên test set. TabPFN chưa có checkpoint trong cache nên bị skip khỏi PR curve. Sau khi tải checkpoint `tabpfn-v2-classifier.ckpt` vào `dog/data/processed/model_cache/tabpfn/` và render lại `dog/src/evaluation/report_eye.py`, PR curve đã có đủ `LR`, `RF`, `MLP`, `MLP (tuned)`, `TabPFN`, `TabICL`, và `TabNet`.

### Q25. Các model cũ trong project có thật sự chạy bằng `.venv` không?

Không thể khẳng định các file kết quả cũ ban đầu được tạo bằng `.venv`, vì chúng đã có sẵn trong project trước khi thêm TabICL. Về mặt code, các script như MLP, TabPFN, TabNet không tự "dùng venv"; chúng dùng môi trường Python nào được gọi để chạy script. Nếu chạy `python dog/src/train/train_eye.py` thì model dùng Python/global environment hiện tại. Nếu chạy `.\.venv\Scripts\python.exe dog/src/train/train_eye.py` thì cùng model đó chạy trong `.venv`. Trong lần thêm TabICL này, TabICL và report render được chạy bằng `.venv` để có đủ package mới và tránh lỗi thiếu dependency. Muốn đồng bộ tuyệt đối, có thể chạy lại tất cả model bằng cùng `.venv`.

### Q26. Trạng thái project sau khi thêm TabICL đang như thế nào?

Về chức năng, project đã ổn: `train_tabicl.py` chạy được, sinh `tabicl_results.json`; `report_eye.py` render lại được report và hai diagram; PR curve và metric bar đều có đủ `LR`, `RF`, `MLP`, `MLP (tuned)`, `TabPFN`, `TabICL`, `TabNet`. Kiểm tra compile Python bằng `.venv\Scripts\python.exe -m compileall -q dog\src` cũng pass. Tuy nhiên git working tree chưa sạch vì có các file đã sửa và file mới chưa commit: README, report, figures, requirements, `report_eye.py`, `train_tabicl.py`, `tabicl_results.json`, và `tabicl_knowledge.md`.

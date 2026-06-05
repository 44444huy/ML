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

### Q27. Code TabICL khác gì TabPFN nếu nhìn khá giống nhau?

Nhìn giống là đúng, vì hai script cố tình dùng cùng protocol đánh giá để so sánh công bằng: cùng đọc `eye_processed.npz`, cùng dùng split, cùng standardize feature bằng train statistics, cùng chạy 5-fold CV, cùng tune threshold theo F1 trên validation, rồi cùng refit trainval và test. Điểm khác chính là model backend trong `fit_predict`: TabPFN gọi `TabPFNClassifier(..., ignore_pretraining_limits=True)`, còn TabICL gọi `TabICLClassifier(...)` với các tham số riêng như `n_estimators`, `batch_size`, `kv_cache`, `allow_auto_download`. Vì vậy code wrapper giống nhau, nhưng thuật toán bên trong khác: TabPFN là Prior-Data Fitted Network cho in-context prediction, còn TabICL dùng kiến trúc column-wise embedding, row-wise interaction, và dataset-wise in-context learning.

### Q28. Cả TabPFN và TabICL đều phải tải model về và chạy bằng `.venv` à?

Nên hiểu thành hai việc riêng. Một là tải/cài thư viện Python: `tabpfn`, `tabicl`, `torch`, `sklearn`, ... được cài vào môi trường Python, trong project này là `.venv`. Hai là tải checkpoint pretrained: cả TabPFN và TabICL đều là foundation models đã pretrained, nên lần đầu chạy cần tải checkpoint nếu cache chưa có. TabPFN hiện đã có checkpoint `tabpfn-v2-classifier.ckpt` trong `dog/data/processed/model_cache/tabpfn/`. TabICL dùng `TabICLClassifier(..., allow_auto_download=True)`, nên khi chạy lần đầu nó tự tải checkpoint TabICL nếu chưa có, thường vào cache mặc định của thư viện/Hugging Face. `.venv` không phải yêu cầu riêng của model; nó chỉ là môi trường mình chọn để đảm bảo đúng dependency. Nếu cài đủ package ở global Python thì cũng chạy được, nhưng dùng `.venv` sạch và dễ tái lập hơn.

### Q29. Checkpoint pretrained từ Hugging Face là gì?

Checkpoint pretrained là file chứa các weights/parameters mà model đã học xong từ trước. Ví dụ TabPFN hoặc TabICL đã được train rất tốn kém trên nhiều tabular tasks trước khi mình dùng. Kết quả của quá trình train đó được lưu thành checkpoint, thường là file `.ckpt`, `.pt`, `.pth`, hoặc `.safetensors`. Khi project gọi `TabPFNClassifier` hay `TabICLClassifier`, thư viện tải checkpoint này, nạp weights vào kiến trúc model, rồi dùng model đã học sẵn để dự đoán trên dog dataset. Hugging Face là một kho lưu trữ phổ biến cho model AI, giống nơi chứa checkpoint, config, tokenizer/metadata và đôi khi cả demo. Trong project này, tải checkpoint không có nghĩa là train lại model; nó chỉ là lấy "bộ não đã học sẵn" về máy để inference.

### Q30. TabPFN cũng dùng checkpoint pretrained như vậy à?

Đúng. TabPFN cũng có phần code thư viện và phần checkpoint pretrained riêng. Code `TabPFNClassifier` định nghĩa cách dựng model và cách chạy `fit/predict_proba`, còn checkpoint chứa weights đã được pretrained trên rất nhiều tabular tasks synthetic. Khi gọi `TabPFNClassifier`, nếu máy chưa có checkpoint thì thư viện tải checkpoint về, trong project này là `dog/data/processed/model_cache/tabpfn/tabpfn-v2-classifier.ckpt`. Sau đó TabPFN dùng checkpoint đã học sẵn để làm in-context learning trên dog train/test context. Nó không gradient-train lại weights chính trên dog dataset.

### Q31. TabNet thì sao, có cần checkpoint pretrained không?

Không. TabNet trong project này không phải foundation model pretrained như TabPFN/TabICL. Nó là một neural network tabular được khởi tạo weights ngẫu nhiên rồi train trực tiếp trên dog dataset bằng gradient descent. Vì vậy TabNet không cần tải checkpoint pretrained từ Hugging Face. Nó chỉ cần cài package `pytorch-tabnet` trong môi trường Python. Khi chạy `train_tabnet.py`, model gọi `TabNetClassifier`, fit trên train fold, early stopping bằng validation fold, dùng class weights để xử lý imbalance, rồi xuất `tabnet_results.json`. Nói ngắn gọn: TabPFN/TabICL = load pretrained checkpoint và làm in-context learning; TabNet = train từ đầu trên dataset hiện tại.

### Q32. Attention của TabNet khác gì attention trong TabPFN và TabICL?

TabNet attention chủ yếu là feature-selection attention: với từng sample và từng decision step, nó tạo một mask trọng số trên các feature/SNP để chọn nên nhìn SNP nào nhiều hơn. Ví dụ ở một step, TabNet có thể gán trọng số cao cho SNP vùng chr18/ALX4 và thấp cho SNP ít liên quan. Mask này nhân trực tiếp với vector feature đầu vào, nên attention của TabNet khá gần với câu hỏi "feature nào quan trọng cho dự đoán này?". TabNet train các mask này từ đầu trên dog dataset bằng gradient descent.

TabPFN/TabICL dùng Transformer attention theo nghĩa rộng hơn: attention là cơ chế để token/sample/context trao đổi thông tin. Với TabPFN, test sample có thể dùng attention để học từ các train samples và label trong context. Với TabICL, attention xuất hiện ở nhiều cấp: column-wise để hiểu phân phối cột, row-wise để SNP trong cùng dog tương tác, và dataset-wise để test dog học từ train dogs có label. Attention ở TabPFN/TabICL không đơn giản là một mask chọn feature; nó tạo contextual embedding và gom thông tin từ context. Vì vậy: TabNet attention dễ hiểu như "chọn SNP nào"; TabPFN/TabICL attention dễ hiểu như "trao đổi/gom thông tin giữa các phần của bảng và giữa các samples".

### Q33. Vì sao SNP trong project được mã hóa thành 0, 1, 2?

Một SNP thường được xét như một vị trí biallelic, ví dụ có hai allele `A/G`. Vì mỗi dog có hai bản sao DNA tại vị trí autosome đó, genotype có thể là `AA`, `AG`, hoặc `GG`. Khi chọn một allele để đếm, thường là alternate allele hoặc minor allele, genotype được đổi thành số lượng allele đó. Ví dụ nếu đếm allele `G`: `AA -> 0`, `AG -> 1`, `GG -> 2`. Vì vậy `0/1/2` là dosage/count của allele được chọn, không phải ba loại SNP khác nhau. Trong bảng ML, mỗi cột SNP chứa các giá trị 0/1/2 qua nhiều dogs; TabICL dùng các cột này để học phân phối cột và tương tác SNP.

### Q34. Chọn allele nào để đếm? Hai allele có phải do hai sợi DNA xoắn kép không?

Allele được đếm là quy ước của dữ liệu/preprocessing. Trong VCF, SNP có `REF` và `ALT`; dosage thường đếm `ALT`, nên `0/0 -> 0`, `0/1 -> 1`, `1/1 -> 2`. Trong PLINK/GWAS, metadata có thể quy ước allele 1/allele 2, minor allele, hoặc effect allele. Nếu đổi allele được đếm thì giá trị đảo lại theo `x -> 2 - x`; ví dụ đếm `G` thì `AA=0, AG=1, GG=2`, còn đếm `A` thì `AA=2, AG=1, GG=0`. Miễn là nhất quán, model vẫn học được pattern, nhưng khi giải thích hệ số/feature importance phải biết đang đếm allele nào.

Hai allele tại một SNP không phải là hai sợi của cùng một DNA double helix. Double helix gồm hai sợi bổ sung nhau, ví dụ `A` đi với `T`, `G` đi với `C`; đó là hai strand của cùng một bản sao DNA. Số hai allele đến từ việc dog là sinh vật diploid: ở autosome, dog có hai chromosome tương đồng, một bản từ bố và một bản từ mẹ. Mỗi bản chromosome có một allele tại cùng vị trí SNP. Nếu bản bố là `A` và bản mẹ là `G`, genotype là `AG`; nếu đang đếm `G`, dosage là `1`.

### Q35. Row-wise interaction trong TabICL khác gì so với TabPFN?

Row-wise interaction là một module được TabICL tách riêng: sau khi tạo cell embeddings theo từng cột, TabICL cho các feature/SNP trong cùng một row attention với nhau để tạo row embedding. Với một dog `[SNP_1, SNP_2, ..., SNP_56]`, bước này học các tương tác ngang như `SNP_1` kết hợp `SNP_2` hoặc nhóm SNP vùng ALX4 kết hợp các SNP khác. TabPFN không nhấn mạnh một bước row-wise interaction riêng như vậy. TabPFN thường mã hóa cả feature vector của một sample thành biểu diễn của sample rồi dùng Transformer/in-context learning để các train/test samples trao đổi thông tin trong context. Nói ngắn gọn: TabICL có pipeline rõ theo bảng `column-wise -> row-wise -> dataset-wise`, còn TabPFN gom feature row vào representation và tập trung hơn vào dataset-wise in-context prediction. Vì vậy TabICL dễ giải thích hơn ở mức "SNP trong cùng dog tương tác thế nào"; TabPFN dễ hiểu hơn như "test sample học từ train samples trong context".

### Q36. `W_Q`, `W_K`, `W_V` là ma trận được pretrain của model à?

Đúng, trong TabICL/TabPFN, `W_Q`, `W_K`, `W_V` là các ma trận weight đã học trong pretraining và được lưu trong checkpoint. Khi chạy trên dog dataset, model không tự tạo mới các ma trận này và cũng không cập nhật chúng bằng gradient; nó load checkpoint rồi dùng các ma trận đó để biến embedding thành query, key, value: `q_i = e_i W_Q`, `k_i = e_i W_K`, `v_i = e_i W_V`. Sau đó attention tính similarity `score_ij = q_i . k_j / sqrt(d)`, softmax thành trọng số, rồi lấy tổng có trọng số các `v_j`. Cần phân biệt: `W_Q/W_K/W_V` là parameters cố định sau pretraining, còn `q/k/v` là vector được tính ra khác nhau cho từng input dog/SNP/context. Trong một Transformer thật còn có nhiều layer và nhiều head, nên không chỉ có một bộ `W_Q/W_K/W_V`, mà có nhiều bộ cho từng layer/head.

## Phụ lục A. TabPFN - thuật toán có minh họa

### A1. TabPFN là gì?

TabPFN là một tabular foundation model cho bài toán phân loại bảng nhỏ. Tên đầy đủ là Prior-Data Fitted Network. Cách hiểu ngắn gọn:

```text
TabPFN = Transformer đã được pretrain để giải nhiều bài toán tabular nhỏ
         bằng cách nhìn train examples trong context và dự đoán test examples.
```

Trong project dog eye color, TabPFN không học lại weights chính trên dog dataset. Nó dùng checkpoint pretrained, rồi nhận:

```text
X_train = SNP vectors của train dogs
y_train = label blue/brown của train dogs
X_test  = SNP vectors của test dogs
```

và xuất:

```text
P(brown), P(blue) cho từng test dog
```

### A2. Trực giác chính: TabPFN học "cách học từ một bảng nhỏ"

MLP học trực tiếp từ dog dataset:

```text
dog train data
    ↓ gradient descent
MLP weights mới
    ↓
predict dog_test
```

TabPFN thì khác:

```text
pretraining trên rất nhiều synthetic tabular tasks
    ↓
checkpoint pretrained
    ↓
nhìn dog train examples trong context
    ↓
predict dog_test
```

Nghĩa là TabPFN học trước một kỹ năng tổng quát: khi thấy một bảng nhỏ gồm feature và label, nó biết cách suy luận label cho row mới.

### A3. Pretraining của TabPFN diễn ra trước project này

Phần này không chạy trong project. Nó đã được tác giả model làm từ trước.

Ý tưởng pretraining:

```text
Lặp rất nhiều lần:
  1. Sinh một bài toán tabular giả lập từ prior
     Ví dụ:
       X = bảng feature giả
       y = label giả được sinh bởi một rule ẩn

  2. Chia thành context và query
       context = vài rows có label
       query   = vài rows bị che label

  3. Đưa context + query vào Transformer

  4. Bắt model dự đoán label của query

  5. Cập nhật weights bằng cross-entropy loss
```

Sơ đồ:

```text
Synthetic task 1:
  row_1: x_1, y_1
  row_2: x_2, y_2
  row_3: x_3, ?
      ↓
  Transformer dự đoán y_3

Synthetic task 2:
  row_1: x_1, y_1
  row_2: x_2, y_2
  row_3: x_3, row_4: ?
      ↓
  Transformer dự đoán y_3, y_4

... lặp rất nhiều task ...
```

Sau pretraining, weights được lưu thành checkpoint. Khi mình dùng TabPFN trong project, mình chỉ tải checkpoint đó về và inference.

### A4. Inference của TabPFN trên dog dataset

Trong project, một fold có dạng:

```text
Train dogs:
  dog_A: [SNP_1=0, SNP_2=2, SNP_3=1, ...], label=brown
  dog_B: [SNP_1=1, SNP_2=2, SNP_3=1, ...], label=blue
  dog_C: [SNP_1=0, SNP_2=1, SNP_3=0, ...], label=brown

Test dog:
  dog_T: [SNP_1=1, SNP_2=2, SNP_3=1, ...], label=?
```

TabPFN nhận toàn bộ context:

```text
[
  (x_A, brown),
  (x_B, blue),
  (x_C, brown),
  (x_T, ?)
]
```

Sau đó Transformer cho `dog_T` nhìn các train dogs:

```text
dog_T attention tới:
  dog_A brown: thấp/vừa
  dog_B blue : cao
  dog_C brown: thấp
```

Output cuối:

```text
P(brown) = 0.25
P(blue)  = 0.75
```

Trong code:

```python
clf = TabPFNClassifier(...)
clf.fit(X_train, y_train)
prob = clf.predict_proba(X_test)[:, 1]
```

`fit` ở đây không giống MLP training. Nó không chạy gradient descent để sửa weights chính của TabPFN trên dog dataset. Nó chủ yếu lưu/chuẩn bị context train để `predict_proba` dùng khi inference.

### A5. Attention trong TabPFN

Với cách nhìn đơn giản, TabPFN biến mỗi row thành biểu diễn vector:

```text
dog_A + label_brown  -> h_A
dog_B + label_blue   -> h_B
dog_C + label_brown  -> h_C
dog_T + label_mask   -> h_T
```

Sau đó attention tính:

```text
q_T = W_Q h_T
k_A = W_K h_A, v_A = W_V h_A
k_B = W_K h_B, v_B = W_V h_B
k_C = W_K h_C, v_C = W_V h_C

score_A = q_T . k_A / sqrt(d)
score_B = q_T . k_B / sqrt(d)
score_C = q_T . k_C / sqrt(d)

alpha = softmax([score_A, score_B, score_C])

context_T = alpha_A * v_A + alpha_B * v_B + alpha_C * v_C
```

Minh họa:

```text
dog_T:
  attention weight tới dog_A brown = 0.15
  attention weight tới dog_B blue  = 0.70
  attention weight tới dog_C brown = 0.15

context_T = 0.15*v_A + 0.70*v_B + 0.15*v_C
```

Quan trọng: đây không phải vote label thô đơn giản. `v_A`, `v_B`, `v_C` là vectors chứa thông tin feature + label + ngữ cảnh đã qua nhiều layer Transformer. Vì vậy TabPFN giống soft k-nearest-neighbor ở trực giác, nhưng thực tế mạnh hơn vì similarity và representation đều được học từ pretraining.

### A6. Pipeline TabPFN trong project

File chính:

```text
dog/src/train/train_tabpfn.py
```

Luồng chạy:

```text
1. Load eye_processed.npz
2. Load train/valid/test splits
3. Với mỗi fold:
     a. Standardize X_train và X_valid bằng statistics của X_train
     b. Gọi TabPFNClassifier
     c. fit(X_train, y_train)
     d. predict_proba(X_valid)
     e. Tune threshold theo F1
     f. Tính PR-AUC, ROC-AUC, F1
4. Aggregate 5-fold CV
5. Refit trên trainval
6. Predict test
7. Ghi tabpfn_results.json
```

Pseudocode:

```text
for fold in folds:
    X_tr, X_va = standardize(X[train], X[valid])

    clf = TabPFNClassifier(pretrained_checkpoint)
    clf.fit(X_tr, y_tr)

    prob_va = clf.predict_proba(X_va)[:, 1]
    threshold = best_f1_threshold(y_va, prob_va)
    metrics = evaluate(y_va, prob_va, threshold)

save metrics
```

### A7. Điểm cần nhớ về TabPFN

```text
TabPFN cần checkpoint pretrained.
TabPFN không train lại weights chính trên dog dataset.
TabPFN dùng train rows + labels như context.
TabPFN attention giúp test row học từ train rows liên quan.
TabPFN wrapper code giống TabICL vì cùng protocol đánh giá.
```

## Phụ lục B. TabNet - thuật toán có minh họa

### B1. TabNet là gì?

TabNet là một neural network cho tabular data. Khác TabPFN/TabICL, TabNet không phải model pretrained trong project này.

```text
TabNet = model train từ đầu trên dog dataset
         + attention mask để chọn feature ở từng decision step
```

Nó giống MLP ở điểm:

```text
weights ban đầu random
    ↓
train bằng gradient descent trên dog dataset
    ↓
weights mới chuyên cho dog eye color
```

Nhưng khác MLP ở điểm: MLP thường dùng toàn bộ feature cùng lúc, còn TabNet học các mask để chọn feature nào nên nhìn ở từng bước.

### B2. Trực giác chính: TabNet ra quyết định theo nhiều bước

Với một dog:

```text
dog_i = [SNP_1, SNP_2, SNP_3, ..., SNP_56]
```

TabNet không chỉ ném toàn bộ vector vào một MLP. Nó làm nhiều decision steps:

```text
Step 1:
  chọn một nhóm SNP quan trọng
  tạo decision vector d1

Step 2:
  nhìn phần thông tin còn lại hoặc nhóm SNP khác
  tạo decision vector d2

Step 3:
  tiếp tục chọn feature và tạo d3

Final:
  d = d1 + d2 + d3
  prediction_head(d) -> P(blue)
```

Minh họa:

```text
Input dog:
  [SNP_1, SNP_2, SNP_chr18_A, SNP_chr18_B, SNP_5, ...]

Step 1 attention mask:
  SNP_chr18_A: 0.70
  SNP_chr18_B: 0.20
  SNP_1      : 0.03
  SNP_2      : 0.02
  others     : nhỏ

Step 2 attention mask:
  SNP_chr18_B: 0.45
  SNP_5      : 0.20
  SNP_9      : 0.10
  others     : nhỏ

Step 3 attention mask:
  nhóm SNP khác hoặc phần residual
```

### B3. Attention mask của TabNet

TabNet tạo một mask `M_t` ở mỗi decision step `t`.

Nếu input có 56 SNP:

```text
x = [x1, x2, x3, ..., x56]
M_t = [m1, m2, m3, ..., m56]
```

Mỗi `m_j` là trọng số cho SNP thứ `j` tại step đó.

TabNet nhân mask với input:

```text
x_masked_t = M_t * x
```

Ví dụ:

```text
x = [SNP_1=0, SNP_2=2, SNP_3=1]

M_1 = [0.05, 0.80, 0.15]

x_masked_1 = [
  0.05 * SNP_1,
  0.80 * SNP_2,
  0.15 * SNP_3
]
```

Nghĩa là ở step 1, model gần như tập trung vào `SNP_2`.

Điểm khác Transformer attention:

```text
Transformer attention:
  token này lấy thông tin từ token khác bằng weighted sum của value vectors.

TabNet attention:
  tạo mask để bật/tắt hoặc tăng/giảm trọng số feature đầu vào.
```

### B4. Các khối chính trong TabNet

TabNet có vài khối quan trọng:

```text
Input features
    ↓
BatchNorm
    ↓
Feature Transformer
    ↓
Attentive Transformer -> tạo mask M_t
    ↓
Masked features M_t * x
    ↓
Feature Transformer -> decision vector d_t
    ↓
Lặp nhiều decision steps
    ↓
Tổng hợp decision vectors
    ↓
Prediction head
```

Sơ đồ đơn giản:

```text
                 ┌─────────────────────┐
Input SNP vector → Feature Transformer  → decision d1
        │        └─────────────────────┘
        │
        ├────→ Attentive Transformer → mask M1 → M1 * input
        │
        ├────→ Attentive Transformer → mask M2 → M2 * input → decision d2
        │
        └────→ Attentive Transformer → mask M3 → M3 * input → decision d3

Final decision = d1 + d2 + d3
Final probability = sigmoid/softmax(Final decision)
```

Trong paper TabNet, mask thường dùng sparse selection, tức là nhiều feature có trọng số gần 0. Nhờ vậy TabNet có thể vừa dự đoán, vừa cho feature importance tương đối dễ đọc.

### B5. Vì sao TabNet có nhiều decision steps?

Một bước attention chỉ có thể chọn một kiểu thông tin nổi bật. Nhiều decision steps cho phép model nhìn bảng theo nhiều lượt:

```text
Step 1: nhìn SNP mạnh nhất vùng ALX4
Step 2: nhìn SNP khác hỗ trợ tín hiệu
Step 3: nhìn pattern còn lại để sửa sai
```

Về trực giác, nó giống quá trình ra quyết định tuần tự:

```text
Tôi nhìn dấu hiệu mạnh nhất trước.
Nếu chưa đủ chắc, tôi nhìn thêm dấu hiệu thứ hai.
Sau đó gom các bằng chứng lại.
```

### B6. Prior/update trong TabNet: tránh nhìn mãi một feature

TabNet có ý tưởng "feature nào đã được dùng nhiều thì lần sau bị giảm ưu tiên". Trực giác:

```text
Ban đầu:
  prior = [1, 1, 1, ..., 1]

Step 1 dùng nhiều SNP_chr18_A
  mask_1[SNP_chr18_A] cao

Sang step 2:
  prior của SNP_chr18_A giảm xuống
  model được khuyến khích nhìn thêm feature khác
```

Không nên hiểu là cấm dùng lại tuyệt đối. Nó chỉ điều chỉnh để các step không bị collapse thành cùng một mask y hệt.

### B7. Training TabNet trong project

File chính:

```text
dog/src/train/train_tabnet.py
```

Luồng chạy:

```text
1. Load eye_processed.npz
2. Load train/valid/test splits
3. Với mỗi fold:
     a. Standardize X_train và X_valid
     b. Khởi tạo TabNetClassifier với weights random
     c. Train trên X_train, y_train
     d. Early stopping bằng validation AUC
     e. Predict probability trên validation
     f. Tính PR-AUC, ROC-AUC, F1
4. Aggregate 5-fold CV
5. Refit trainval với internal validation
6. Predict test
7. Ghi tabnet_results.json
```

Pseudocode:

```text
for fold in folds:
    X_tr, X_va = standardize(X[train], X[valid])

    clf = TabNetClassifier(random_initial_weights)
    clf.fit(
        X_train=X_tr,
        y_train=y_tr,
        eval_set=[(X_va, y_va)],
        weights={0: 1.0, 1: n_neg / n_pos},
        early_stopping=True,
    )

    prob_va = clf.predict_proba(X_va)[:, 1]
    metrics = evaluate(y_va, prob_va, threshold=0.5)

save metrics
```

### B8. TabNet xử lý imbalance như thế nào?

Dog eye color bị imbalance mạnh:

```text
brown: rất nhiều
blue : khoảng 4%
```

Nếu không xử lý, model dễ học cách đoán brown gần như mọi lúc.

Trong project, TabNet dùng class weights:

```python
weights = {0: 1.0, 1: n_neg / n_pos}
```

Nghĩa là lỗi trên class blue được phạt nặng hơn lỗi trên class brown. Ý tưởng này giống `pos_weight` của MLP, nhưng được đưa vào API `pytorch-tabnet`.

### B9. TabNet attention khác TabPFN/TabICL attention bằng ví dụ

Giả sử có test dog:

```text
dog_T = [SNP_1=0, SNP_2=2, SNP_ALX4=1, SNP_4=0, ...]
```

TabNet hỏi:

```text
"Trong 56 SNP của chính dog_T, SNP nào nên được nhìn nhiều?"
```

Minh họa:

```text
TabNet step 1 mask:
  SNP_ALX4: 0.82
  SNP_2   : 0.10
  SNP_1   : 0.02
  SNP_4   : 0.01
```

TabPFN/TabICL hỏi:

```text
"dog_T nên học từ train dog nào trong context?"
"cell/feature/row này nên lấy thông tin từ token nào?"
```

Minh họa:

```text
TabPFN/TabICL dataset attention:
  dog_B blue : 0.60
  dog_D blue : 0.20
  dog_A brown: 0.15
  dog_C brown: 0.05
```

Vậy:

```text
TabNet attention = chọn feature trong một sample.
TabPFN/TabICL attention = gom thông tin giữa tokens/rows/samples trong context.
```

### B10. Điểm cần nhớ về TabNet

```text
TabNet không cần checkpoint pretrained.
TabNet train từ đầu trên dog dataset.
TabNet attention tạo mask feature theo từng decision step.
TabNet có thể giải thích feature importance tự nhiên hơn MLP.
TabNet giống MLP ở chỗ phải gradient-train trên dog data.
TabNet khác TabPFN/TabICL ở chỗ không làm in-context learning từ checkpoint pretrained.
```

## Phụ lục C. Bảng so sánh nhanh TabPFN, TabICL, TabNet

| Điểm so sánh | TabPFN | TabICL | TabNet |
|---|---|---|---|
| Có pretrained checkpoint? | Có | Có | Không trong project này |
| Có train lại weights chính trên dog dataset? | Không | Không | Có |
| Cách học trên dog dataset | In-context learning | In-context learning | Gradient descent |
| Attention chính dùng để làm gì? | Gom thông tin từ context train/test | Column-wise, row-wise, dataset-wise context | Chọn feature/SNP bằng mask |
| Code project giống ai? | Giống TabICL ở wrapper đánh giá | Giống TabPFN ở wrapper đánh giá | Giống MLP hơn vì phải train |
| Output trong project | `tabpfn_results.json` | `tabicl_results.json` | `tabnet_results.json` |

Một câu nhớ nhanh:

```text
TabPFN: model pretrained học cách giải bài toán tabular nhỏ bằng context.
TabICL: model pretrained mã hóa bảng theo cột, row, rồi context.
TabNet: model train từ đầu, dùng attention mask để chọn SNP quan trọng.
```

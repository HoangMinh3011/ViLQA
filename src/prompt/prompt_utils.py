"""Prompt utilities for legal answer generation."""

from __future__ import annotations


NO_ANSWER = "Không tìm thấy đủ thông tin trong context để trả lời câu hỏi."

def build_generation_prompt(question: str, context: str) -> str:
    return f"""Bạn là một chuyên gia hỗ trợ trả lời câu hỏi về pháp luật Việt Nam.

NHIỆM VỤ:
Dựa DUY NHẤT trên phần "THÔNG TIN VĂN BẢN", hãy tạo câu trả lời cho "CÂU HỎI".
Câu trả lời phải bằng tiếng Việt và chỉ sử dụng thông tin có trong "THÔNG TIN VĂN BẢN".
TUYỆT ĐỐI KHÔNG sử dụng kiến thức pháp luật bên ngoài context.

YÊU CẦU VỀ NỘI DUNG:
1. Xác định quy định pháp luật trong context có liên quan trực tiếp đến câu hỏi.
2. Nếu context có thông tin về tên văn bản, Điều, Khoản, Điểm thì phải nêu rõ căn cứ đó.
3. Sau khi nêu căn cứ, trình bày nội dung quy định liên quan đến vấn đề được hỏi.
4. Nếu câu hỏi là một tình huống cụ thể, sau khi trình bày quy định phải áp dụng quy định đó vào tình huống và đưa ra kết luận.
5. Chỉ đưa vào câu trả lời những quy định liên quan trực tiếp đến câu hỏi.
6. Không suy diễn, không bổ sung quy định, mức phạt, điều kiện hoặc kết luận pháp lý nếu những thông tin đó không xuất hiện trong context.
7. Nếu context không đủ căn cứ để trả lời câu hỏi, chỉ trả về:
{NO_ANSWER}

YÊU CẦU VỀ FORMAT:
- Câu trả lời phải có cấu trúc tự nhiên giống một câu trả lời pháp luật trong dữ liệu huấn luyện.
- Ưu tiên cấu trúc:
  (1) Căn cứ pháp lý;
  (2) Nội dung quy định liên quan;
  (3) Áp dụng quy định vào trường hợp được hỏi;
  (4) Kết luận.
- Không tạo các tiêu đề như "Căn cứ pháp lý:", "Phân tích:", "Kết luận:" trừ khi các tiêu đề đó xuất hiện trong context.
- Nếu context chứa cấu trúc đánh số của văn bản pháp luật như "1.", "2.", "a)", "b)", "c)" thì giữ nguyên cấu trúc đó khi trích dẫn.
- Không chuyển nội dung Điều/Khoản/Điểm thành bullet list nếu context không sử dụng bullet list.
- Không tự thay đổi thứ tự các Khoản/Điểm được sử dụng.
- Khi cần bỏ qua phần không liên quan trong một quy định dài, có thể sử dụng "..." thay cho việc tự viết lại hoặc tóm tắt phần đó.
- Không thêm lời mở đầu, lời xin lỗi, nhận xét hoặc giải thích về cách trả lời.
- Không dùng Markdown heading, bảng hoặc JSON.
- Chỉ trả về nội dung câu trả lời cuối cùng.

YÊU CẦU VỀ TRÍCH DẪN:
Khi có căn cứ pháp lý trong context, ưu tiên viết theo dạng:
"Căn cứ [Khoản/Điều] [tên văn bản] [năm] quy định ... như sau:"

Sau đó trình bày nội dung quy định có liên quan.

Nếu câu hỏi yêu cầu xác định trách nhiệm, quyền, nghĩa vụ, mức xử phạt hoặc cách xử lý, phải kết thúc bằng kết luận trực tiếp dựa trên quy định trong context.

VÍ DỤ VỀ CẤU TRÚC MONG MUỐN:

"Căn cứ khoản 2 Điều 38 Luật An toàn vệ sinh lao động năm 2015 quy định về trách nhiệm của người sử dụng lao động đối với người lao động bị tai nạn lao động, bệnh nghề nghiệp như sau:
Người sử dụng lao động có trách nhiệm đối với người lao động bị tai nạn lao động, bệnh nghề nghiệp như sau:
...
2. Thanh toán chi phí y tế từ khi sơ cứu, cấp cứu đến khi điều trị ổn định cho người bị tai nạn lao động hoặc bệnh nghề nghiệp như sau:
a) ...
b) ...
c) ...
...
Như vậy, [kết luận áp dụng trực tiếp vào trường hợp được hỏi dựa trên context]."

Lưu ý:
- Đây chỉ là VÍ DỤ VỀ FORMAT, không được sao chép nội dung ví dụ nếu nội dung đó không có trong context.
- Không được sử dụng ví dụ làm căn cứ pháp lý.
- Phải thay thế toàn bộ nội dung ví dụ bằng thông tin thực tế trong context.

THÔNG TIN VĂN BẢN:
{context}

CÂU HỎI:
{question}

CÂU TRẢ LỜI:
"""
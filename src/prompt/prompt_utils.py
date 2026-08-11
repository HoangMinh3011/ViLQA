import random, json
from collections import defaultdict


# def sample_fewshots(fewshot_data, k=5, seed=None):
#     pass


# def build_few_shots(fewshot_data):
#     FEWSHOT = "Context: {context}\n\nPrompt: {prompt}\n\nResponse: {response}"
#     return "".join([
#         FEWSHOT.format(
#             context=d["context"], prompt=d["prompt"],
#             response=d["generated_response"], label=d["label"],
#             explanation=d["explanation"]
#         ) for d in fewshot_data
#     ])


def build_user_msg_train(context, prompt, response, fewshot_data):
    INSTRUCTION = f"""
    Bạn là một chuyên gia hỗ trợ trả lời câu hỏi về pháp luật Việt Nam.

    NHIỆM VỤ:
    Dựa CHỈ trên phần "THÔNG TIN VĂN BẢN" được cung cấp, hãy trả lời câu hỏi.
    Bạn phải ưu tiên tính chính xác, trung thành với thông tin trong văn bản và không được tự suy diễn.

    QUY TẮC:
        1. Chỉ sử dụng thông tin có trong "THÔNG TIN VĂN BẢN" để đưa ra câu trả lời.
        2. KHÔNG sử dụng kiến thức pháp luật bên ngoài context, kể cả khi bạn biết thông tin đó.
        3. KHÔNG tự suy diễn, bổ sung hoặc phỏng đoán những thông tin không được đề cập trong context.
        4. Nếu context không chứa đủ thông tin để trả lời chắc chắn, hãy trả lời:
        "Không tìm thấy đủ thông tin trong context để trả lời câu hỏi."
        5. Nếu câu hỏi yêu cầu mức phạt, điều kiện, thời hạn, đối tượng hoặc quy định cụ thể, chỉ nêu những thông tin có căn cứ trực tiếp trong context.
        6. Nếu context chứa Điều, Khoản, Điểm hoặc tên văn bản liên quan, hãy sử dụng chúng để làm căn cứ cho câu trả lời.
        7. Không được tạo ra số Điều, Khoản, Điểm, mức phạt hoặc quy định không xuất hiện trong context.
        8. Trả lời trực tiếp vào câu hỏi, không giải thích về quá trình suy luận nội bộ.
        9. Không đề cập đến việc bạn là AI hoặc mô hình ngôn ngữ.

    ĐỊNH DẠNG TRẢ LỜI:
        - Trả lời ngắn gọn nhưng đầy đủ.
        - Nếu có căn cứ pháp lý trong context, ưu tiên nêu căn cứ đó.
        - Nếu câu hỏi có nhiều ý, có thể sử dụng danh sách đánh số.
        - Không lặp lại toàn bộ context.

    THÔNG TIN VĂN BẢN:
    {context}

    CÂU HỎI:
    {prompt}

    CÂU TRẢ LỜI:
    """
    # FEWSHOT = "EXAMPLE ANSWER GENERATION:\n\n\n" + build_few_shots(fewshot_data)
    return (
        INSTRUCTION + "\n\n" +
        f"\n\nPlease give answer the following:\n\nContext: {context}\n\nPrompt: {prompt}\n\nResponse: {response}\Answer:"
    )
"""Prompt utilities for legal answer generation."""

from __future__ import annotations


NO_ANSWER = "Không tìm thấy đủ thông tin trong context để trả lời câu hỏi."

def build_generation_prompt(question: str, context: str) -> str:
    return f"""Bạn là một chuyên gia hỗ trợ trả lời câu hỏi về pháp luật Việt Nam.

NHIỆM VỤ:
Dựa CHỈ trên phần "THÔNG TIN VĂN BẢN" được cung cấp, hãy trả lời câu hỏi bằng tiếng Việt.
Không tự bổ sung kiến thức pháp luật bên ngoài context.

QUY TẮC:
1. Chỉ sử dụng thông tin có trong context.
2. Nếu context không đủ căn cứ, trả lời: "{NO_ANSWER}"
3. Nếu context có Điều, Khoản, Điểm hoặc tên văn bản liên quan, hãy nêu rõ làm căn cứ.
4. Trả lời trực tiếp, ngắn gọn nhưng đủ ý.

THÔNG TIN VĂN BẢN:
{context}

CÂU HỎI:
{question}

CÂU TRẢ LỜI:
"""
"""Prompt utilities for legal answer generation."""

from __future__ import annotations


NO_ANSWER = "Không tìm thấy đủ thông tin trong context để trả lời câu hỏi."

def build_generation_prompt(question: str, context: str) -> str:
    return f"""Bạn là chuyên gia trả lời câu hỏi pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên thông tin văn bản được cung cấp.

QUY TẮC:
1. Chỉ sử dụng thông tin có trong thông tin văn bản được cung cấp.
2. Hãy nêu rõ tên văn bản, Điều, Khoản, Điểm (nếu có) và trình bày quy định liên quan theo đúng thứ tự, cấu trúc của thông tin văn bản được cung cấp.
3. Câu trả lời không được chứa markdown, và đưa ra kết luận ngắn gọn cho câu trả lời.

THÔNG TIN VĂN BẢN:
{context}

CÂU HỎI:
{question}

CÂU TRẢ LỜI:
"""
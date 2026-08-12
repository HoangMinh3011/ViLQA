"""Prompt utilities for legal answer generation."""

from __future__ import annotations


NO_ANSWER = "Khong tim thay du thong tin trong context de tra loi cau hoi."


def build_generation_prompt(question: str, context: str) -> str:
    return f"""Ban la mot chuyen gia ho tro tra loi cau hoi ve phap luat Viet Nam.

NHIEM VU:
Dua CHI tren phan "THONG TIN VAN BAN" duoc cung cap, hay tra loi cau hoi bang tieng Viet.
Khong tu bo sung kien thuc phap luat ben ngoai context.

QUY TAC:
1. Chi su dung thong tin co trong context.
2. Neu context khong du can cu, tra loi: "{NO_ANSWER}"
3. Neu context co Dieu, Khoan, Diem hoac ten van ban lien quan, hay neu ro lam can cu.
4. Tra loi truc tiep, ngan gon nhung du y.

THONG TIN VAN BAN:
{context}

CAU HOI:
{question}

CAU TRA LOI:
"""

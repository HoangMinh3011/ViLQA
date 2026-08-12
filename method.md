# Phương pháp đề xuất: Hybrid Retrieval + Cross-Encoder Reranking cho Legal QA

## 1. Tổng quan

Pipeline gồm 5 giai đoạn chính:

1. **Tiền xử lý**: segment văn bản pháp luật thành các chunk có kích thước cố định, có overlap giữa các chunk liên tiếp.
2. **Retrieval**: kết hợp BM25 (lexical) và Biencoder (dense) để lấy top-k chunk ứng viên cho mỗi câu hỏi.
3. **Reranking**: dùng cross-encoder để chấm điểm lại độ liên quan giữa câu hỏi và từng chunk ứng viên, chọn ra top-k evidence chính xác nhất.
4. **Context Reconstruction**: map top-k evidence trở lại văn bản gốc để khôi phục ngữ cảnh đầy đủ xung quanh (tránh mất ngữ cảnh do chunk bị cắt cụt).
5. **Generation**: đưa context đã khôi phục + câu hỏi vào LLM để sinh câu trả lời cuối cùng.

Thiết kế này áp dụng **đồng nhất cho mọi loại văn bản pháp luật** (Luật, Nghị định, Thông tư, Quyết định, TCVN/Quy chuẩn kỹ thuật...) vì chunking dựa trên kích thước cố định, không phụ thuộc vào việc văn bản có cấu trúc Chương/Điều/Khoản/Điểm rõ ràng hay không (nhiều Thông tư và hầu hết TCVN không có cấu trúc này).

## 2. Sơ đồ kiến trúc

```text
Legal Corpus
     |
     v
Document Normalize
     |
     v
Segment + Chunking (overlap ~50)
     |
     v
Question --+--> Biencoder --+
            |                +--> Retrieval (Hybrid)
            +--> BM25 -------+
                              |
                              v
                      Candidate Documents
                              |
                              v
                       Candidate Chunks (top-k)
                              |
                              v
                     Rerank (Cross-Encoder)
                              |
                              v
                       Top-k Evidence
                              |
                              v
              Context Reconstruction (map lại corpus gốc)
                              |
                              v
                             LLM
                              |
                              v
                           Answer
```

## 3. Tiền xử lý: Document Normalize

Corpus thật (dạng `{"id", "name", "link", "passage"}`) cần chuẩn hóa trước khi chunking — loại noise ngắt dòng và trích metadata cơ bản (không bắt buộc, chỉ hỗ trợ hiển thị/debug):

```python
import re

DOC_NUMBER_PATTERN = re.compile(r"Số:\s*([\w\d\/\-]+)")
DOC_TITLE_PATTERN = re.compile(
    r"(?im)^\s*(THÔNG TƯ|NGHỊ ĐỊNH|QUYẾT ĐỊNH|LUẬT|NGHỊ QUYẾT|CHỈ THỊ|TCVN|QCVN)\b.*$"
)

def normalize_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_document_metadata(raw_text: str) -> dict:
    number_match = DOC_NUMBER_PATTERN.search(raw_text)
    title_match = DOC_TITLE_PATTERN.search(raw_text)
    return {
        "document_number": number_match.group(1) if number_match else None,
        "title": title_match.group(0).strip() if title_match else None,
    }

def normalize_document(raw: dict) -> dict:
    text = normalize_text(raw["passage"])
    meta = extract_document_metadata(raw["passage"])
    return {
        "document_id": str(raw["id"]),
        "title": meta["title"],
        "text": text,
        "metadata": {
            "document_number": meta["document_number"],
            "source": raw.get("link"),
        },
    }
```

## 4. Tiền xử lý: Segment + Chunking (overlap)

Chunking theo **token cố định**, dùng chính tokenizer của LLM (hoặc Biencoder) để đếm — không đếm theo từ/ký tự, đảm bảo nhất quán với ngân sách token dùng ở bước Context Reconstruction về sau.

```python
def chunk_document(document: dict, tokenizer, chunk_size=300, overlap=50):
    text = document["text"]
    tokens = tokenizer.tokenize(text.encode("utf-8"))

    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.detokenize(chunk_tokens).decode("utf-8", errors="ignore")

        chunks.append({
            "chunk_id": f"{document['document_id']}_chunk_{idx}",
            "document_id": document["document_id"],
            "chunk_index": idx,        # thứ tự chunk trong văn bản
            "token_start": start,       # offset token trong văn bản gốc — dùng ở bước 6
            "token_end": end,
            "text": chunk_text,
        })

        if end == len(tokens):
            break
        start += chunk_size - overlap
        idx += 1

    return chunks


def chunk_corpus(documents: list[dict], tokenizer, chunk_size=300, overlap=50):
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, tokenizer, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
```

**Chọn `chunk_size`/`overlap`:**
- `chunk_size` phải nhỏ hơn `max_seq_length` của Biencoder — cần kiểm tra giá trị thật của model trước khi chốt số, tránh bị cắt mất nội dung khi encode.
- `overlap=50` (đề xuất ban đầu, ~15–20% nếu `chunk_size≈300`) giúp câu quan trọng nằm vắt ngang ranh giới 2 chunk vẫn xuất hiện trọn vẹn trong ít nhất một chunk.
- Cả hai tham số cần tuning trên validation set bằng Recall@k, không chỉ chọn cố định theo cảm tính.

## 5. Retrieval: Hybrid (BM25 + Biencoder)

### 5.1. Biencoder (Dense)

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

encoder = SentenceTransformer("huyydangg/DEk21_hcmute_embedding_v2")

def encode_documents(texts, batch_size=32):
    return encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True)

def build_faiss_index(embeddings):
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def dense_retrieve(query, encoder, index, chunks, top_k=20):
    q = encoder.encode(query, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q.reshape(1, -1), top_k)
    return [{**chunks[i], "dense_score": float(s)} for s, i in zip(scores[0], indices[0])]
```

### 5.2. BM25 (`bm25s`)

```python
# pip install bm25s --break-system-packages
import bm25s
# ⚠️ Cần word-segmentation tiếng Việt thật (underthesea/pyvi/VnCoreNLP) trước khi tokenize,
# không dùng .split() thuần túy.

def build_bm25(chunks: list[dict]):
    corpus_texts = [c["text"] for c in chunks]
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords=None)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever

def bm25_retrieve(query, bm25, chunks, top_k=20):
    query_tokens = bm25s.tokenize([query])
    results, scores = bm25.retrieve(query_tokens, k=top_k)
    return [{**chunks[i], "bm25_score": float(s)} for i, s in zip(results[0], scores[0])]
```

### 5.3. Hybrid Fusion → top-k Candidate Chunks

Chuẩn hóa min-max rồi weighted sum — không cộng thẳng raw score vì thang điểm BM25 và cosine similarity khác nhau.

```python
def minmax_normalize(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)

def hybrid_fusion(bm25_results, dense_results, alpha=0.5):
    candidates = {}
    bm25_norm = minmax_normalize([r["bm25_score"] for r in bm25_results])
    dense_norm = minmax_normalize([r["dense_score"] for r in dense_results])

    for item, s in zip(bm25_results, bm25_norm):
        c = candidates.setdefault(item["chunk_id"], item.copy())
        c["bm25_norm"] = float(s)
    for item, s in zip(dense_results, dense_norm):
        c = candidates.setdefault(item["chunk_id"], item.copy())
        c["dense_norm"] = float(s)

    for c in candidates.values():
        c["hybrid_score"] = alpha * c.get("bm25_norm", 0.0) + (1 - alpha) * c.get("dense_norm", 0.0)
    return sorted(candidates.values(), key=lambda x: x["hybrid_score"], reverse=True)


def get_candidate_chunks(question, bm25, encoder, dense_index, chunks, top_k=20, alpha=0.5):
    bm25_results = bm25_retrieve(question, bm25, chunks, top_k=top_k)
    dense_results = dense_retrieve(question, encoder, dense_index, chunks, top_k=top_k)
    fused = hybrid_fusion(bm25_results, dense_results, alpha=alpha)
    return fused[:top_k]
```

`alpha` cần tuning trên validation set — không cố định 0.5.

## 6. Reranking: Cross-Encoder → Top-k Evidence

Cross-encoder chấm điểm trực tiếp từng cặp `(question, chunk_text)`, chính xác hơn nhiều so với việc chỉ dựa vào hybrid score (vốn tính riêng biệt từng chunk, không "đọc" cả câu hỏi và chunk cùng lúc).

Model gợi ý cho tiếng Việt: **`AITeamVN/Vietnamese_Reranker`** (fine-tune từ `bge-reranker-v2-m3`, huấn luyện trên ~1.1 triệu triplet tiếng Việt, có `sentence-transformers` `CrossEncoder` wrapper sẵn). Lựa chọn để so sánh thêm: `itdainb/PhoRanker`, `ViRanker` (arXiv:2509.09131).

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("AITeamVN/Vietnamese_Reranker", max_length=512)

def rerank(question: str, candidate_chunks: list[dict]) -> list[dict]:
    pairs = [(question, item["text"]) for item in candidate_chunks]
    scores = reranker.predict(pairs)  # tự batch nội bộ

    for item, score in zip(candidate_chunks, scores):
        item["rerank_score"] = float(score)

    return sorted(candidate_chunks, key=lambda x: x["rerank_score"], reverse=True)


def select_top_k_evidence(ranked_chunks: list[dict], k=5) -> list[dict]:
    return ranked_chunks[:k]
```

⚠️ **Lưu ý hiệu năng**: cross-encoder tốn compute hơn nhiều so với hybrid score (mỗi cặp `(question, chunk)` là một forward pass transformer). Không nên rerank trực tiếp trên toàn corpus — đây là lý do vẫn cần bước Hybrid Retrieval lọc trước xuống `top_k` (mục 5) rồi mới rerank trên tập nhỏ đó. Nên đo thời gian rerank/câu hỏi thực tế trên GPU trước khi chạy full evaluation.

> Ghi chú từ model card `AITeamVN/Vietnamese_Reranker`: điểm số trên domain pháp lý có giảm nhẹ so với domain tổng quát. Nên đo Recall@k/nDCG@k thực tế trên câu hỏi pháp luật của bạn, và so sánh với `PhoRanker`/`ViRanker` nếu kết quả không như mong đợi.

## 7. Context Reconstruction: map lại corpus gốc

Vì chunk được cắt theo token cố định (có thể cắt cụt giữa câu/giữa ý), bước này **mở rộng theo offset token trong văn bản gốc** — lấy thêm một khoảng token trước/sau chunk trong cùng văn bản để khôi phục ngữ cảnh, rồi cắt theo ngân sách token thực tế của LLM trước khi đưa vào prompt.

```python
def reconstruct_context(evidence: list[dict], document_lookup: dict, tokenizer,
                         max_context_tokens=5000, expand_window=50) -> str:
    seen_ids = set()
    blocks = []
    used_tokens = 0

    for item in evidence:  # đã sort theo rerank_score giảm dần
        chunk_id = item["chunk_id"]
        if chunk_id in seen_ids:
            continue

        document = document_lookup[item["document_id"]]
        doc_tokens = tokenizer.tokenize(document["text"].encode("utf-8"))

        start = max(0, item["token_start"] - expand_window)
        end = min(len(doc_tokens), item["token_end"] + expand_window)
        expanded_tokens = doc_tokens[start:end]
        n_tokens = len(expanded_tokens)

        if used_tokens + n_tokens > max_context_tokens:
            if not blocks:
                # Evidence quan trọng nhất đã quá dài -> cắt bớt, không bỏ trắng context
                expanded_tokens = expanded_tokens[:max_context_tokens]
                text = tokenizer.detokenize(expanded_tokens).decode("utf-8", errors="ignore")
                blocks.append(text)
            break  # bỏ evidence rank thấp hơn khi ngân sách token không đủ

        seen_ids.add(chunk_id)
        text = tokenizer.detokenize(expanded_tokens).decode("utf-8", errors="ignore")
        blocks.append(text)
        used_tokens += n_tokens

    return "\n\n---\n\n".join(blocks)
```

`document_lookup`: dict `{document_id: normalized_document}` — cần giữ lại toàn văn từng văn bản gốc (không chỉ chunk) để tra cứu ở bước này.

## 8. Generation: LLM → Answer

```python
from llama_cpp import Llama

llm = Llama(
    model_path="PATH_TO_QWEN_GGUF",
    n_ctx=8192,
    n_threads=8,
    n_gpu_layers=-1,
    verbose=False,
)

def build_prompt(question: str, evidence_text: str) -> str:
    return f"""
Bạn là trợ lý trả lời câu hỏi pháp luật Việt Nam.

Hãy trả lời câu hỏi dựa trên các căn cứ pháp luật được cung cấp.
Không tự bổ sung thông tin pháp luật không xuất hiện trong context.

[CĂN CỨ PHÁP LUẬT]
{evidence_text}

[CÂU HỎI]
{question}

[YÊU CẦU]
1. Trả lời trực tiếp câu hỏi.
2. Chỉ sử dụng thông tin có căn cứ trong context.
3. Nếu có thể, nêu rõ Điều, Khoản, Điểm làm căn cứ (nếu context có ghi rõ).
4. Nếu context không đủ để kết luận, hãy nói rõ rằng không đủ căn cứ.
"""

def generate_answer(llm, question: str, evidence_text: str) -> str:
    prompt = build_prompt(question, evidence_text)
    output = llm(prompt, max_tokens=1024, temperature=0.1, top_p=0.9, stop=["[END]"])
    return output["choices"][0]["text"].strip()
```

## 9. Hàm tổng hợp end-to-end

```python
def retrieve_evidence(question, bm25, encoder, dense_index, chunks,
                       retrieval_k=20, evidence_k=5, alpha=0.5):
    # Question -> Candidate Chunks (Hybrid Retrieval)
    candidate_chunks = get_candidate_chunks(
        question, bm25, encoder, dense_index, chunks, top_k=retrieval_k, alpha=alpha,
    )

    # Candidate Chunks -> Rerank (Cross-Encoder) -> Top-k Evidence
    ranked = rerank(question, candidate_chunks)
    return select_top_k_evidence(ranked, k=evidence_k)


def answer_question(question, bm25, encoder, dense_index, chunks, document_lookup, tokenizer, llm,
                     max_context_tokens=5000):
    evidence = retrieve_evidence(question, bm25, encoder, dense_index, chunks)

    # Top-k Evidence -> Context Reconstruction
    context_text = reconstruct_context(evidence, document_lookup, tokenizer, max_context_tokens=max_context_tokens)

    # Context -> LLM -> Answer
    answer = generate_answer(llm, question, context_text)

    return {"question": question, "evidence": evidence, "context": context_text, "answer": answer}
```

## 10. Build Index (offline — chạy một lần, tách khỏi Inference)

```python
def build_indexes(raw_documents: list[dict], tokenizer, chunk_size=300, overlap=50):
    documents = [normalize_document(raw) for raw in raw_documents]
    document_lookup = {d["document_id"]: d for d in documents}

    chunks = chunk_corpus(documents, tokenizer, chunk_size=chunk_size, overlap=overlap)

    texts = [c["text"] for c in chunks]
    embeddings = encode_documents(texts)
    dense_index = build_faiss_index(embeddings)

    bm25 = build_bm25(chunks)

    return {
        "chunks": chunks,
        "document_lookup": document_lookup,
        "dense_index": dense_index,
        "bm25": bm25,
    }


def save_artifacts(artifacts: dict, out_dir="/kaggle/working/artifacts"):
    import os, pickle, faiss
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/chunks.pkl", "wb") as f:
        pickle.dump(artifacts["chunks"], f)
    with open(f"{out_dir}/document_lookup.pkl", "wb") as f:
        pickle.dump(artifacts["document_lookup"], f)

    artifacts["bm25"].save(f"{out_dir}/bm25_index")
    faiss.write_index(artifacts["dense_index"], f"{out_dir}/dense.index")


def load_artifacts(artifact_dir: str):
    import pickle, faiss, bm25s

    with open(f"{artifact_dir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(f"{artifact_dir}/document_lookup.pkl", "rb") as f:
        document_lookup = pickle.load(f)

    bm25 = bm25s.BM25.load(f"{artifact_dir}/bm25_index")
    dense_index = faiss.read_index(f"{artifact_dir}/dense.index")

    return chunks, document_lookup, bm25, dense_index
```

Quy trình khuyến nghị trên Kaggle: chạy `build_indexes()` + `save_artifacts()` trong một notebook riêng (Notebook A), publish output thành Kaggle Dataset, rồi notebook inference/eval (Notebook B) chỉ cần `load_artifacts()` — tránh phải rebuild embedding/BM25 mỗi lần chạy, giảm rủi ro mất session giữa chừng.

## 11. Hyperparameter khởi điểm

```python
CHUNK_SIZE = 300          # token/chunk — cần < max_seq_length của Biencoder
CHUNK_OVERLAP = 50        # token overlap giữa 2 chunk liên tiếp
RETRIEVAL_K = 20          # top-k candidate chunks trước rerank
EVIDENCE_K = 5             # top-k evidence cuối cùng (cap trên; ngân sách token ở mục 7 là giới hạn thật)
ALPHA_HYBRID = 0.5        # trọng số BM25 vs Dense trong hybrid fusion
MAX_CONTEXT_TOKENS = 5000  # ngân sách token cho Context Reconstruction
EXPAND_WINDOW = 50         # số token mở rộng mỗi phía khi map evidence về văn bản gốc
```

Tất cả cần tuning trên validation set; test set giữ độc lập hoàn toàn trong suốt quá trình chọn hyperparameter.

## 12. Checklist implement theo thứ tự

1. **Document Normalize** — chạy trên toàn corpus, kiểm tra tỷ lệ trích được `title`/`document_number` (không cần 100%, chỉ là metadata phụ trợ).
2. **Segment + Chunking** — xác nhận `max_seq_length` thật của Biencoder trước khi chốt `CHUNK_SIZE`; kiểm tra chunk đầu/cuối văn bản, văn bản rất ngắn/rất dài để chắc không lỗi off-by-one ở overlap.
3. **Dense Retrieval + BM25** — build index riêng từng kênh, đo Recall@k độc lập.
4. **Hybrid Fusion** — so sánh với từng kênh riêng lẻ, tune `alpha`.
5. **Cross-Encoder Rerank** — load `AITeamVN/Vietnamese_Reranker`, đo thời gian rerank/câu hỏi thực tế trên GPU; đo Recall@k/nDCG@k sau rerank so với trước.
6. **Context Reconstruction** — kiểm tra không vượt `n_ctx`, không lỗi khi offset âm/vượt biên văn bản.
7. **Tách Notebook Build Index / Inference** — publish artifacts thành Kaggle Dataset trước khi eval end-to-end.
8. **LLM (Qwen3.5-4B GGUF, frozen)** — ghép toàn bộ, đánh giá answer (EM/F1/BERTScore/LLM-as-a-Judge/Citation Correctness).
9. **Ablation** — so sánh: BM25-only / Dense-only / Hybrid / Hybrid+Cross-Encoder Rerank (full pipeline).

## 13. Điểm cần xác nhận trước khi code chính thức

- Kiểm tra thực tế `encoder.max_seq_length` của `huyydangg/DEk21_hcmute_embedding_v2` trước khi chốt `CHUNK_SIZE`.
- Xác nhận lại tên checkpoint chính xác của model sinh câu trả lời (`unsloth/Qwen3.5-4B-GGUF`) trên HuggingFace/unsloth.
- So sánh nhanh giữa `AITeamVN/Vietnamese_Reranker`, `itdainb/PhoRanker`, `ViRanker` trên một tập validation nhỏ có silver-label (rút từ câu trả lời trong `train.json` bằng cách trích citation Điều/Nghị định trong `answer` rồi match với corpus) — chọn checkpoint có Recall@k/nDCG@k tốt nhất trên domain pháp luật.
- Đo thời gian cross-encoder rerank thực tế trên GPU Kaggle (T4/P100) để ước lượng tổng thời gian chạy full evaluation set trước khi submit.
- Thống kê phân phối độ dài văn bản trong corpus (số token/document) để chọn `CHUNK_SIZE`/`CHUNK_OVERLAP` phù hợp thực tế, không chỉ dựa trên cảm tính.

## 14. Hạn chế và hướng phát triển tiếp theo

- **Không khai thác cấu trúc pháp lý** (Chương/Điều/Khoản/Điểm): pipeline hiện tại coi mọi văn bản như văn bản phẳng, chia theo token — đơn giản, ổn định, chạy được trên mọi loại văn bản, nhưng có thể kém chính xác hơn một pipeline biết tận dụng ranh giới Điều/Khoản thật khi văn bản có cấu trúc rõ ràng.
- **Hướng cải tiến tiếp theo**: sau khi baseline này chạy ổn định và có số liệu rõ ràng, có thể bổ sung một nhánh xử lý riêng cho các văn bản có cấu trúc rõ ràng (Luật/Nghị định) — parse theo Điều/Khoản/Điểm, xây graph phân cấp, dùng Graph Expansion thay cho việc chỉ dựa vào token liền kề — rồi so sánh trực tiếp với baseline hiện tại để đo lường đóng góp thực sự của việc khai thác cấu trúc.

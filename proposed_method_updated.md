# Phương pháp đề xuất: Hybrid Retrieval + Hierarchical Legal Graph cho Legal QA

> File này mô tả **pipeline end-to-end cuối cùng**, khớp với sơ đồ kiến trúc đã chốt.
> Mục tiêu: đủ chi tiết để implement trực tiếp (vibe coding) — mỗi node trong sơ đồ tương ứng
> với một module/hàm cụ thể bên dưới.

> **Cập nhật (v1.1) — tối ưu hiệu năng cho corpus 1M+ node trên Kaggle:**
> - Không dùng Query Rewriting (giữ pipeline gọn, đúng quyết định đã chốt).
> - Giữ nguyên thứ tự Rerank **trước** khi vào LLM (không theo hướng LLM-answer-based reranking của paper tham khảo).
> - Thay `networkx.DiGraph` bằng **dict-based tree** (mục 3.4) — build/tra cứu O(1)-O(n) thay vì có overhead lớn ở quy mô 1M+ node.
> - Thay `rank_bm25` bằng **`bm25s`** (mục 4.2) — nhanh hơn nhiều lần ở quy mô lớn vì dùng numpy/scipy backend thay vì linear scan thuần Python.
> - Thêm **Context Reconstruction theo ngân sách token thực tế** (mục 8) — tránh lỗi vượt `n_ctx` của LLM.
> - Thêm mục 11 **tách rõ Notebook Build Index / Notebook Inference** — giải quyết vấn đề mất session giữa chừng trên Kaggle.

---

## 1. Sơ đồ kiến trúc (chốt cuối)

```text
Legal Corpus
     |
     v
Legal Structure Parsing
     |
     v
Hierarchical Knowledge
     |
     v
Graph Construction ------------------------------+
     ^                                            |
     |                                            | (Graph Expansion)
     |                                            v
Question --+--> Biencoder --+                     |
            |                +--> Retrieval       |
            +--> BM25 -------+                     |
                              |                     |
                              v                     |
                      Candidate Documents           |
                              |                     |
                              v                     |
                       Candidate Chunks ------------+
                              |
                              v
                      Reference Chunks
                              |
                              v
                          Evidence
                              |
                              v
                           Rerank
                              |
                              v
                           Chunks
                              |
                              v
                       Orginal Chunks
                              |
                              v
                             LLM
                              |
                              v
                           Answer
```

Không có đường tắt nào bỏ qua Graph Expansion / Rerank — **mọi câu hỏi đều đi qua toàn bộ chuỗi này**. Đây là full pipeline duy nhất (không phải sơ đồ ablation).

---

## 2. Quy ước đặt tên (naming) theo sơ đồ

Để tránh nhầm lẫn khi code, quy ước ý nghĩa từng node như sau:

| Node trong sơ đồ | Ý nghĩa kỹ thuật | Kiểu dữ liệu |
|---|---|---|
| `Candidate Documents` | Văn bản pháp luật ứng viên sau hybrid retrieval ở cấp document | `List[dict]` |
| `Candidate Chunks` | Legal chunks (node phân cấp) ứng viên sau hybrid retrieval ở cấp chunk | `List[Node]` |
| `Reference Chunks` | Candidate Chunks đã được mở rộng qua Graph (thêm cha/con/anh em) | `List[Node]` |
| `Evidence` | Reference Chunks đã được **gán điểm liên quan** (chưa sắp xếp/cắt) | `List[ScoredNode]` |
| `Rerank` | Bước **sắp xếp** Evidence theo `final_score` | hàm, không phải data |
| `Chunks` | Top-k node sau khi rerank (evidence cuối cùng, đã chọn lọc) | `List[ScoredNode]` |
| `Orginal Chunks` | Nội dung pháp luật gốc được khôi phục đầy đủ (context reconstruction) từ `Chunks` | `str` (context text) |

> Lưu ý: `Orginal Chunks` **không phải** là chunk thô chưa lọc — nó là bước **khôi phục văn bản gốc/đầy đủ ngữ cảnh** (Điều → Khoản → Điểm) tương ứng với các `Chunks` đã được chọn, để đưa vào LLM dưới dạng dễ đọc thay vì các đoạn `embedding_text` rời rạc.

---

## 3. Tiền xử lý: Legal Corpus → Graph Construction

### 3.1. Legal Corpus

```python
document = {
    "document_id": "doc_001",
    "title": "Tên văn bản",
    "text": "...",
    "metadata": {
        "document_number": "...",
        "source": "...",
        "effective_date": "...",
    },
}
```

### 3.2. Legal Structure Parsing (Regex)

```python
import re

PATTERNS = {
    "chapter": re.compile(r"(?im)^\s*CHƯƠNG\s+([IVXLCDM]+)\.?\s*(.*?)\s*$"),
    "section": re.compile(r"(?im)^\s*MỤC\s+(\d+)\.?\s*(.*?)\s*$"),
    "article": re.compile(r"(?im)^\s*Điều\s+(\d+)\.?\s*(.*?)\s*$"),
    "clause":  re.compile(r"(?m)^\s*(\d+)\.\s+(.*)$"),
    "point":   re.compile(r"(?m)^\s*([a-zđ])\)\s+(.*)$"),
}
```

Output: danh sách node phân cấp (`Văn bản → Chương → Mục → Điều → Khoản → Điểm`), không tạo node giả cho cấp không tồn tại.

```python
node = {
    "node_id": "doc001_ch02_sec01_art07_c01_p_a",
    "node_type": "point",
    "document_id": "doc001",
    "chapter_number": "II", "chapter_title": "...",
    "section_number": "1",  "section_title": "...",
    "article_number": 7,    "article_title": "...",
    "clause_number": 1,
    "point_number": "a",
    "text": "...",
    "parent_id": "doc001_ch02_sec01_art07_c01",
    "path": ["doc001", "chapter_II", "section_1", "article_7", "clause_1", "point_a"],
}
```

⚠️ **Cần QA thủ công** stage này trên một tập mẫu (vd. 20–30 văn bản đa dạng thể loại) trước khi build Graph — lỗi parser sẽ lan truyền xuống toàn bộ pipeline.

### 3.3. Hierarchical Knowledge → embedding_text

Mỗi node có 2 representation:
- `node["text"]`: nội dung gốc.
- `node["embedding_text"]`: nội dung có thêm path phân cấp, dùng cho dense retrieval.

```python
def build_embedding_text(node: dict) -> str:
    parts = []
    if node.get("document_title"):
        parts.append(f"[Văn bản] {node['document_title']}")
    if node.get("chapter_title"):
        parts.append(f"[Chương {node['chapter_number']}] {node['chapter_title']}")
    if node.get("section_title"):
        parts.append(f"[Mục {node['section_number']}] {node['section_title']}")
    if node.get("article_number") is not None:
        article = f"[Điều {node['article_number']}]"
        if node.get("article_title"):
            article += f" {node['article_title']}"
        parts.append(article)
    if node.get("clause_number") is not None:
        parts.append(f"[Khoản {node['clause_number']}]")
    if node.get("point_number"):
        parts.append(f"[Điểm {node['point_number']}]")
    parts.append(node["text"])
    return "\n".join(parts)
```

### 3.4. Graph Construction (dict-based tree — thay cho networkx)

⚠️ **Đổi từ `networkx.DiGraph` sang dict thuần**: cấu trúc pháp luật (Văn bản→Chương→Mục→Điều→Khoản→Điểm) là một **cây**, mỗi node chỉ có đúng 1 cha. Với 1M+ node, `networkx` (thuần Python, nhiều lớp object) build/tra cứu chậm hơn đáng kể so với dict thuần. Dùng dict-based tree giúp build O(n) một lần, tra cứu parent/children O(1).

```python
def build_tree_index(nodes: list[dict]) -> dict:
    children_map = {}   # parent_id -> [child_id, ...]
    parent_map = {}      # node_id -> parent_id
    node_lookup = {}      # node_id -> node dict đầy đủ

    for node in nodes:
        node_id = node["node_id"]
        node_lookup[node_id] = node
        parent_id = node.get("parent_id")
        parent_map[node_id] = parent_id
        if parent_id:
            children_map.setdefault(parent_id, []).append(node_id)

    return {"children": children_map, "parent": parent_map, "lookup": node_lookup}
```

Quan hệ tối thiểu (v1): quan hệ cha-con ngầm định qua `parent_id` (tương đương `HAS_CHILD`/`HAS_CHAPTER/HAS_SECTION/HAS_ARTICLE/HAS_CLAUSE/HAS_POINT`).
Mở rộng sau (v2 — nếu cần quan hệ không phải cây, ví dụ `REFERENCES`, `AMENDS`, `REPEALS`, `DEFINES`): lưu riêng thành một dict `{node_id: [related_node_id, ...]}` bổ sung, không cần quay lại `networkx` trừ khi thực sự cần visualize/graph algorithm phức tạp (shortest path, centrality...).

---

## 4. Nhánh Question → Candidate Chunks

### 4.1. Biencoder (Dense Retrieval)

Model: `huyydangg/DEk21_hcmute_embedding_v2`

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("huyydangg/DEk21_hcmute_embedding_v2")

def encode_documents(texts, batch_size=32):
    return encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True)

def encode_query(query):
    return encoder.encode(query, normalize_embeddings=True)
```

FAISS index:

```python
import faiss, numpy as np

def build_faiss_index(embeddings):
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def dense_retrieve(query, encoder, index, nodes, top_k=20):
    q = encoder.encode(query, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q.reshape(1, -1), top_k)
    return [
        {**nodes[idx], "dense_score": float(score)}
        for score, idx in zip(scores[0], indices[0])
    ]
```

### 4.2. BM25 (Lexical Retrieval)

⚠️ **Đổi từ `rank_bm25` sang `bm25s`**: `rank_bm25.get_scores()` là linear scan qua toàn bộ corpus cho **mỗi query** (thuần Python) — ở quy mô 1M+ node, đây là nguyên nhân chính khiến retrieval mỗi câu hỏi rất chậm. `bm25s` dùng numpy/scipy backend, nhanh hơn hàng chục–hàng trăm lần ở quy mô lớn.

```python
# pip install bm25s --break-system-packages
import bm25s
# ⚠️ Vẫn cần word-segmentation tiếng Việt thật (underthesea/pyvi/VnCoreNLP)
# trước khi đưa vào bm25s.tokenize, KHÔNG dùng .split() thuần túy.

def build_bm25(nodes: list[dict]):
    corpus_texts = [n["text"] for n in nodes]
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords=None)  # thay bằng tokenizer tiếng Việt nếu cần
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever

def bm25_retrieve(query, bm25, nodes, top_k=20):
    query_tokens = bm25s.tokenize([query])
    results, scores = bm25.retrieve(query_tokens, k=top_k)
    return [{**nodes[idx], "bm25_score": float(score)} for idx, score in zip(results[0], scores[0])]
```

`bm25s` hỗ trợ `retriever.save(path)` / `bm25s.BM25.load(path)` để lưu/tải index trực tiếp — dùng trong bước tách Notebook Build Index / Inference ở mục 11.

### 4.3. Retrieval → Candidate Documents

Hybrid fusion (min-max normalize rồi weighted sum, không cộng raw score):

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
        c = candidates.setdefault(item["node_id"], item.copy())
        c["bm25_norm"] = float(s)
    for item, s in zip(dense_results, dense_norm):
        c = candidates.setdefault(item["node_id"], item.copy())
        c["dense_norm"] = float(s)

    for c in candidates.values():
        c["hybrid_score"] = (
            alpha * c.get("bm25_norm", 0.0)
            + (1 - alpha) * c.get("dense_norm", 0.0)
        )
    return sorted(candidates.values(), key=lambda x: x["hybrid_score"], reverse=True)
```

`alpha` cần tuning trên validation set (không cố định 0.5).

Nếu corpus lớn, thực hiện retrieval 2 tầng: `Candidate Documents` (lọc văn bản trước) → rồi mới `Candidate Chunks` (retrieval trong phạm vi các văn bản ứng viên). Nếu corpus nhỏ/vừa, có thể retrieval thẳng ở cấp chunk.

### 4.4. Candidate Chunks

```python
def get_candidate_chunks(question, bm25, encoder, dense_index, nodes, top_k=20, alpha=0.5):
    bm25_results = bm25_retrieve(question, bm25, nodes, top_k=top_k)
    dense_results = dense_retrieve(question, encoder, dense_index, nodes, top_k=top_k)
    fused = hybrid_fusion(bm25_results, dense_results, alpha=alpha)
    return fused[:top_k]
```

---

## 5. Graph Expansion: Candidate Chunks → Reference Chunks

Dùng `tree` (dict-based, mục 3.4) thay vì `networkx` — tra cứu parent/children/sibling đều O(1), quan trọng khi mỗi câu hỏi đều gọi hàm này.

```python
def graph_expand(
    tree, seed_nodes,
    include_parent=True,
    include_children=True,
    include_siblings=True,
    child_depth=1,
):
    children_map, parent_map, lookup = tree["children"], tree["parent"], tree["lookup"]
    expanded = {}

    for node in seed_nodes:
        node_id = node["node_id"]
        expanded[node_id] = {**node, "expansion_type": "direct"}

        parent_id = parent_map.get(node_id)

        # Node cha
        if include_parent and parent_id:
            expanded.setdefault(parent_id, {
                **lookup[parent_id], "node_id": parent_id, "expansion_type": "parent",
            })

        # Node con (BFS tới child_depth)
        if include_children:
            frontier, visited = [node_id], {node_id}
            for _ in range(child_depth):
                next_frontier = []
                for current in frontier:
                    for child_id in children_map.get(current, []):
                        if child_id in visited:
                            continue
                        visited.add(child_id)
                        expanded.setdefault(child_id, {
                            **lookup[child_id], "node_id": child_id, "expansion_type": "child",
                        })
                        next_frontier.append(child_id)
                frontier = next_frontier

        # Node anh em (cùng cha)
        if include_siblings and parent_id:
            for sib_id in children_map.get(parent_id, []):
                if sib_id == node_id or sib_id in expanded:
                    continue
                expanded[sib_id] = {
                    **lookup[sib_id], "node_id": sib_id, "expansion_type": "sibling",
                }

    return list(expanded.values())
```

→ Output: **Reference Chunks**. Đây chưa phải evidence cuối — gần nhau trong graph không đồng nghĩa với liên quan tới câu hỏi.

---

## 6. Reference Chunks → Evidence (gán điểm liên quan)

`Evidence` = Reference Chunks đã được **gán điểm** kết hợp giữa relevance (hybrid score) và graph prior — **chưa sắp xếp/cắt**.

```python
GRAPH_PRIOR = {
    "direct":    1.0,
    "parent":    0.8,
    "child":     0.8,
    "sibling":   0.6,
    "reference": 0.7,  # dành cho quan hệ REFERENCES (v2)
}

def score_evidence(reference_chunks, alpha=0.8, beta=0.2):
    evidence = []
    for chunk in reference_chunks:
        hybrid_score = chunk.get("hybrid_score", 0.0)
        expansion_type = chunk.get("expansion_type", "direct")
        graph_score = GRAPH_PRIOR.get(expansion_type, 0.5)

        item = chunk.copy()
        item["graph_score"] = graph_score
        item["final_score"] = alpha * hybrid_score + beta * graph_score
        evidence.append(item)
    return evidence
```

> Ghi chú thiết kế: các node được thêm vào chỉ qua Graph Expansion (parent/child/sibling) không có `hybrid_score` gốc (vì chưa từng được BM25/Dense retrieval trực tiếp) → mặc định `hybrid_score = 0.0`, điểm số của chúng chủ yếu đến từ `graph_score`. Đây là điểm cần lưu ý khi tuning `alpha`/`beta`.

---

## 7. Evidence → Rerank → Chunks

`Rerank` chỉ là bước **sắp xếp** Evidence theo `final_score` (v1 dùng heuristic tuyến tính ở trên). `Chunks` là top-k sau khi cắt.

```python
def rerank(evidence: list[dict]) -> list[dict]:
    return sorted(evidence, key=lambda x: x["final_score"], reverse=True)

def select_top_k_chunks(ranked_evidence, k=5):
    return ranked_evidence[:k]
```

> **Nâng cấp v2 (đề xuất, không bắt buộc cho bản đầu):** thay heuristic tuyến tính bằng một reranker học được (cross-encoder tiếng Việt, hoặc mono-T5) để so sánh ablation `heuristic rerank` vs `learned rerank`.

---

## 8. Chunks → Orginal Chunks (Context Reconstruction theo ngân sách token)

⚠️ **Fix lỗi vượt `n_ctx`**: đặt `evidence_k` cố định (vd. 5) không đảm bảo tổng context luôn vừa `n_ctx`, vì có câu hỏi Graph Expansion trả về các Điều rất dài. Thay vì cắt theo **số lượng** chunk, cắt theo **ngân sách token thực tế** (đếm bằng tokenizer của chính LLM), ưu tiên evidence rank cao trước (vì `chunks` đã được sort theo `final_score` giảm dần ở bước Rerank):

```python
def reconstruct_context(chunks: list[dict], node_lookup: dict, llm, max_context_tokens=5000) -> str:
    # max_context_tokens nên nhỏ hơn n_ctx, chừa chỗ cho prompt template + câu hỏi + max_tokens sinh ra
    seen_ids = set()
    blocks = []
    used_tokens = 0

    for item in chunks:  # đã sort theo final_score giảm dần từ bước rerank
        node_id = item["node_id"]
        if node_id in seen_ids:
            continue

        node = node_lookup[node_id]
        text = node.get("embedding_text", node["text"])
        n_tokens = len(llm.tokenize(text.encode("utf-8")))

        if used_tokens + n_tokens > max_context_tokens:
            if not blocks:
                # Trường hợp hiếm: evidence quan trọng nhất đã quá dài -> cắt bớt thay vì bỏ trắng context
                truncated_ids = llm.tokenize(text.encode("utf-8"))[:max_context_tokens]
                text = llm.detokenize(truncated_ids).decode("utf-8", errors="ignore")
                blocks.append(text)
            break  # bỏ các evidence rank thấp hơn khi ngân sách token không đủ

        seen_ids.add(node_id)
        blocks.append(text)
        used_tokens += n_tokens

    return "\n\n".join(blocks)
```

> v2 nâng cao: khi nhiều `Chunks` cùng thuộc một Điều (vd. Điểm a, b của cùng Khoản 1), nên gộp lại thành một block "Điều X → Khoản 1 → a) ... b) ..." thay vì lặp lại tiêu đề Điều/Khoản nhiều lần — vừa tiết kiệm ngân sách token, vừa tự nhiên hơn khi đọc.

---

## 9. Orginal Chunks → LLM → Answer

Model: `unsloth/Qwen3.5-4B-GGUF` — **giữ frozen** ở giai đoạn đầu để tách bạch đóng góp của retrieval.

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
3. Nếu có thể, nêu rõ Điều, Khoản, Điểm làm căn cứ.
4. Nếu context không đủ để kết luận, hãy nói rõ rằng không đủ căn cứ.
"""

def generate_answer(llm, question: str, evidence_text: str) -> str:
    prompt = build_prompt(question, evidence_text)
    output = llm(prompt, max_tokens=1024, temperature=0.1, top_p=0.9, stop=["[END]"])
    return output["choices"][0]["text"].strip()
```

---

## 10. Hàm tổng hợp end-to-end

```python
def retrieve_evidence(
    question, bm25, encoder, dense_index, nodes, tree,
    retrieval_k=20, evidence_k=5, alpha=0.5,
):
    # Question -> Candidate Chunks
    candidate_chunks = get_candidate_chunks(
        question, bm25, encoder, dense_index, nodes,
        top_k=retrieval_k, alpha=alpha,
    )

    # Candidate Chunks -> Reference Chunks (Graph Expansion, dict-tree)
    reference_chunks = graph_expand(
        tree, candidate_chunks,
        include_parent=True, include_children=True,
        include_siblings=True, child_depth=1,
    )

    # Reference Chunks -> Evidence
    evidence = score_evidence(reference_chunks)

    # Evidence -> Rerank -> Chunks
    ranked = rerank(evidence)
    chunks = select_top_k_chunks(ranked, k=evidence_k)  # evidence_k chỉ là cap trên, ngân sách token mới là giới hạn thật (mục 8)

    return chunks


def answer_question(question, bm25, encoder, dense_index, nodes, tree, node_lookup, llm, max_context_tokens=5000):
    chunks = retrieve_evidence(question, bm25, encoder, dense_index, nodes, tree)

    # Chunks -> Orginal Chunks (Context Reconstruction theo ngân sách token)
    context_text = reconstruct_context(chunks, node_lookup, llm, max_context_tokens=max_context_tokens)

    # Orginal Chunks -> LLM -> Answer
    answer = generate_answer(llm, question, context_text)

    return {
        "question": question,
        "chunks": chunks,
        "context": context_text,
        "answer": answer,
    }
```

---

## 11. Tách Notebook Build Index / Notebook Inference (bắt buộc trên Kaggle với 1M+ node)

⚠️ **Nguyên nhân chính gây mất session trên Kaggle**: nếu mỗi lần chạy pipeline đều rebuild lại toàn bộ index (parse corpus → build tree → encode 1M+ embeddings → build BM25) trước khi trả lời được câu hỏi nào, sẽ rất dễ vượt quá thời gian/RAM cho phép của session, đặc biệt khi phải chạy lại nhiều lần để debug. Giải pháp: build index **một lần duy nhất**, publish thành Kaggle Dataset, notebook inference chỉ cần load lại.

### 11.1. Notebook A — Build Index (chạy 1 lần, sau đó Save Version → publish Dataset)

```python
def build_indexes(documents: list[dict]):
    nodes = parse_legal_documents(documents)          # Legal Structure Parsing

    for node in nodes:                                 # Hierarchical Knowledge
        node["embedding_text"] = build_embedding_text(node)

    tree = build_tree_index(nodes)                       # Graph Construction (dict-tree)
    node_lookup = tree["lookup"]

    # Cân nhắc: chỉ embed node lá (Điểm/Khoản) để giảm số lượng embedding cần tính;
    # node cha (Điều/Chương) chỉ cần giữ trong `tree` để phục vụ Graph Expansion, không cần embed riêng.
    leaf_nodes = [n for n in nodes if n["node_type"] in ("point", "clause")]
    texts = [n["embedding_text"] for n in leaf_nodes]
    embeddings = encode_documents(texts)
    dense_index = build_faiss_index(embeddings)

    bm25 = build_bm25(leaf_nodes)

    return {
        "nodes": leaf_nodes,
        "tree": tree,
        "node_lookup": node_lookup,
        "dense_index": dense_index,
        "bm25": bm25,
    }


def save_artifacts(artifacts: dict, out_dir="/kaggle/working/artifacts"):
    import os, pickle, faiss

    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/nodes.pkl", "wb") as f:
        pickle.dump(artifacts["nodes"], f)
    with open(f"{out_dir}/tree.pkl", "wb") as f:
        pickle.dump(artifacts["tree"], f)
    with open(f"{out_dir}/node_lookup.pkl", "wb") as f:
        pickle.dump(artifacts["node_lookup"], f)

    artifacts["bm25"].save(f"{out_dir}/bm25_index")
    faiss.write_index(artifacts["dense_index"], f"{out_dir}/dense.index")
```

Sau khi chạy xong: **Save Version** notebook này → vào tab Output → **"Create Dataset"** từ thư mục `artifacts/` → publish thành Kaggle Dataset riêng (vd. `legalqa-artifacts-v1`).

### 11.2. Notebook B — Inference / Eval (Add Data → import Dataset từ Notebook A)

```python
def load_artifacts(artifact_dir="/kaggle/input/legalqa-artifacts-v1"):
    import pickle, faiss, bm25s

    with open(f"{artifact_dir}/nodes.pkl", "rb") as f:
        nodes = pickle.load(f)
    with open(f"{artifact_dir}/tree.pkl", "rb") as f:
        tree = pickle.load(f)
    with open(f"{artifact_dir}/node_lookup.pkl", "rb") as f:
        node_lookup = pickle.load(f)

    bm25 = bm25s.BM25.load(f"{artifact_dir}/bm25_index")
    dense_index = faiss.read_index(f"{artifact_dir}/dense.index")

    return nodes, tree, node_lookup, bm25, dense_index
```

Notebook B chỉ cần `load_artifacts()` rồi chạy thẳng `answer_question(...)` — không bao giờ phải rebuild tree/embedding/BM25 lại nữa, kể cả khi session bị ngắt giữa chừng. Đây là quy trình bắt buộc khi corpus ở quy mô 1M+ node.

---

## 12. Hyperparameter khởi điểm

```python
RETRIEVAL_K = 20        # top-k candidate chunks trước graph expansion
EVIDENCE_K  = 5          # top-k chunks cuối cùng đưa vào LLM
ALPHA_HYBRID = 0.5       # trọng số BM25 vs Dense
GRAPH_CHILD_DEPTH = 1    # độ sâu mở rộng node con
RERANK_ALPHA = 0.8       # trọng số hybrid_score trong final_score
RERANK_BETA  = 0.2       # trọng số graph_score trong final_score
```

Tất cả cần tuning trên validation set; test set giữ độc lập hoàn toàn trong suốt quá trình chọn hyperparameter.

---

## 13. Checklist implement theo đúng thứ tự (khuyến nghị cho vibe coding)

1. **Legal Structure Parsing** — parse thử 20–30 văn bản, kiểm tra thủ công 100% cấu trúc trước khi đi tiếp.
2. **Graph Construction (dict-tree)** — build `tree` bằng dict thuần (mục 3.4), test tra cứu parent/children trên vài văn bản để kiểm tra hierarchy đúng. Không dùng networkx.
3. **Dense Retrieval (Biencoder)** — chỉ embed node lá (Điểm/Khoản), build FAISS, đo Recall@k độc lập.
4. **BM25 (`bm25s`)** — build index bằng `bm25s` (nhớ dùng tokenizer tiếng Việt thật), đo Recall@k độc lập, đo thời gian retrieval/query để chắc chắn đủ nhanh ở quy mô full corpus.
5. **Hybrid Fusion** — kết hợp BM25 + Dense, so sánh với từng kênh riêng lẻ.
6. **Graph Expansion** — thêm parent/child/sibling qua `tree`, kiểm tra qua vài case thủ công (không để sibling bị bỏ sót như bản nháp trước).
7. **Evidence scoring + Rerank** — implement `score_evidence` + `rerank`, đo Recall@k / MRR / nDCG@k sau rerank so với trước.
8. **Context Reconstruction theo ngân sách token** — kiểm tra output không bị trùng lặp và không vượt `n_ctx` ngay cả với câu hỏi có Graph Expansion trả về nhiều node.
9. **Tách Notebook Build Index / Notebook Inference** — publish artifacts thành Kaggle Dataset trước khi chạy eval end-to-end (mục 11), tránh mất session giữa chừng.
10. **LLM (Qwen3.5-4B GGUF, frozen)** — ghép toàn bộ, đánh giá answer (Exact Match / F1 / BERTScore / LLM-as-a-Judge / Citation Correctness).
11. **Ablation** — so sánh: BM25-only / Dense-only / Hybrid / Hybrid+Graph / Full (Hybrid+Graph+Rerank), tất cả dùng chung một Qwen frozen.

---

## 14. Điểm cần xác nhận trước khi code chính thức

- Xác nhận lại tên checkpoint chính xác của model sinh câu trả lời (`unsloth/Qwen3.5-4B-GGUF`) trên HuggingFace/unsloth trước khi hard-code vào script.
- ✅ Đã chốt: chỉ embed/BM25 trên **node lá** (Điểm/Khoản); node cha (Điều/Chương) chỉ giữ trong `tree` phục vụ Graph Expansion, không embed riêng — tránh trùng lặp nội dung cạnh tranh trong candidate chunks và giảm số lượng embedding cần tính.
- Chọn công cụ tokenization tiếng Việt cụ thể cho BM25 (underthesea/pyvi/VnCoreNLP) và dùng nhất quán cho cả bước build corpus lẫn query-time.
- ✅ Đã chốt: không dùng Query Rewriting; giữ Rerank trước LLM (không theo hướng LLM-answer-based reranking).
- Cần benchmark thời gian build index + build embedding trên tập full corpus (1M+ node) ngay từ đầu để ước lượng số Kaggle session cần dùng cho Notebook A, tránh bị timeout giữa chừng khi encode embedding.
- Cân nhắc dùng GPU accelerator trên Kaggle cho bước `encode_documents()` — encode 1M+ đoạn văn bản bằng CPU sẽ rất chậm; nên bật GPU T4/P100 riêng cho Notebook A.
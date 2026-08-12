# VLQA Legal QA Pipeline

Pipeline này implement theo hướng trong `method.md`: normalize corpus pháp luật, chia chunk có overlap, hybrid retrieval bằng BM25 + biencoder, rerank bằng cross-encoder, reconstruct context từ văn bản gốc, sau đó đưa context vào LLM GGUF để sinh câu trả lời.

## Cấu trúc chính

```text
config.py                    # default config cho model và pipeline
scripts/
  build_index.py             # offline: build BM25 + dense FAISS artifacts
  run_qa.py                  # online: retrieval, rerank, LLM generation
  download_model.py          # tải file GGUF của LLM
src/
  prepare_data/              # load, normalize, chunk corpus
  retrieval/                 # BM25, biencoder, hybrid fusion
  rerank/                    # cross-encoder reranking
  generation/                # reconstruct context
  model/                     # llama.cpp GGUF backend
  pipeline/                  # offline và online pipeline
  io/                        # save/load artifacts
selected-contexts/           # corpus văn bản pháp luật dạng JSON
artifacts/                   # index sinh ra sau khi build
models/                      # file .gguf của LLM
```

## Cài đặt

```bash
pip install -r requirements.txt
```

Nếu chạy trên GPU CUDA với `llama-cpp-python`, hãy cài wheel/phiên bản phù hợp CUDA của môi trường đang dùng. Biencoder và reranker sẽ tự dùng CUDA nếu `torch` nhận GPU.

## Chuẩn bị corpus

Thư mục `selected-contexts` là corpus được dùng để build retrieval index. Mỗi file JSON nên có các field như:

```json
{
  "id": "context_100050",
  "name": "Tên văn bản",
  "link": "https://...",
  "passage": "Nội dung văn bản..."
}
```

Trên Kaggle, có thể upload `selected-contexts.zip` rồi giải nén:

```bash
unzip /kaggle/input/<dataset-name>/selected-contexts.zip -d /kaggle/working/VLQA
```

Miễn là sau khi giải nén có đường dẫn `/kaggle/working/VLQA/selected-contexts`.

## Tải LLM GGUF

Model mặc định: `unsloth/Qwen3.5-4B-GGUF`, file `Qwen3.5-4B-UD-Q4_K_XL.gguf`.

```bash
python scripts/download_model.py \
  --repo-id unsloth/Qwen3.5-4B-GGUF \
  --filename Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --output models/Qwen3.5-4B-UD-Q4_K_XL.gguf
```

## Build index

```bash
python scripts/build_index.py \
  --contexts-dir selected-contexts \
  --artifacts-dir artifacts \
  --biencoder-model NghiemAbe/Vi-Legal-Bi-Encoder-v2 \
  --chunk-size 300 \
  --chunk-overlap 50 \
  --dense-batch-size 32
```

Tham số hay dùng:

- `--limit`: build thử với một số lượng file nhỏ.
- `--skip-dense`: chỉ build BM25, nhanh hơn nhưng retrieval yếu hơn.
- `--tokenizer-model`: tokenizer dùng để chia chunk; mặc định dùng biencoder model.

Artifacts tạo ra gồm `documents.json`, `chunks.json`, `document_lookup.pkl`, `bm25.pkl`, `dense.index`, `embeddings.npy`.

## Chạy QA

Chạy một câu hỏi:

```bash
python scripts/run_qa.py \
  --question "Hợp đồng đã công chứng có được huỷ bỏ không?" \
  --artifacts-dir artifacts \
  --llm-model-path models/Qwen3.5-4B-UD-Q4_K_XL.gguf
```

Chạy file câu hỏi và xuất submission:

```bash
python scripts/run_qa.py \
  --questions-file data/public-official.json \
  --artifacts-dir artifacts \
  --llm-model-path models/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --output data/submission.json \
  --output-format submission
```

Tạo file zip nộp bài:

```bash
python scripts/run_qa.py \
  --questions-file data/public-official.json \
  --artifacts-dir artifacts \
  --llm-model-path models/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --output data/submission.json \
  --output-format submission \
  --zip-output data/submission.zip
```

## Điều chỉnh context để tránh vượt context window

Nếu gặp lỗi dạng `Requested tokens exceed context window`, giảm các tham số sau:

```bash
python scripts/run_qa.py \
  --questions-file data/public-official.json \
  --artifacts-dir artifacts \
  --llm-model-path models/Qwen3.5-4B-UD-Q4_K_XL.gguf \
  --retrieval-k 10 \
  --evidence-k 3 \
  --max-context-tokens 3500 \
  --expand-window 20 \
  --llm-n-ctx 8192 \
  --llm-max-tokens 768 \
  --output data/submission.json \
  --output-format submission
```

Ý nghĩa nhanh:

- `--retrieval-k`: số candidate chunks lấy từ hybrid retrieval.
- `--evidence-k`: số evidence chunks sau rerank đưa vào context reconstruction.
- `--max-context-tokens`: ngân sách token tối đa cho context.
- `--expand-window`: số token mở rộng quanh chunk evidence trong văn bản gốc.
- `--llm-n-ctx`: context window của llama.cpp.
- `--llm-max-tokens`: số token tối đa LLM sinh ra.

## Inspect retrieval

Dùng `--inspect` để xem candidate, evidence và context preview mà không gọi LLM:

```bash
python scripts/run_qa.py \
  --question "Điều kiện huỷ bỏ hợp đồng công chứng là gì?" \
  --artifacts-dir artifacts \
  --inspect \
  --evidence-k 3
```
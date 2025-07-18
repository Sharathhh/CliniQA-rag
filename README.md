## CLINIQA-RAG

Build a Retrieval-Augmented Generation (RAG) pipeline that enables document-based question answering using open-source components like FAISS, sentence-transformers, and Hugging Face LLMs.

## Tech Stacks Used

| Component        | Tool/Model                                |
| ---------------- | ----------------------------------------- |
| Embeddings       | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector Store     | `FAISS` (Local, fast similarity search)   |
| LLM (Q\&A)       | `google/flan-t5-base` via Hugging Face    |
| Framework        | `LangChain` for pipeline orchestration    |
| Document Parsing | `PyPDFLoader` from LangChain              |
| Chunking         | `RecursiveCharacterTextSplitter`          |


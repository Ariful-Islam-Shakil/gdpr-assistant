# GDPR Intelligence Engine: Advanced RAG Pipeline

A state-of-the-art compliance intelligence system designed to transform raw GDPR legal documentation into a high-performance **Retrieval-Augmented Generation (RAG)** environment. This project implements a sophisticated multi-stage pipeline that combines semantic understanding with keyword precision to deliver expert-level legal assistance.

## 🌟 Key Features

- **Multi-Stage Retrieval Engine**:
    - **Stage 1**: Hybrid search (FAISS + BM25) across the entire GDPR Article corpus.
    - **Stage 2**: Dynamic extraction of legally associated Recitals for retrieved Articles.
    - **Stage 3**: Secondary refinement search across candidate Recitals to select the most impactful context.
- **Intelligent Article-Recital Mapping**: Automatically crawls and syncs legal relationships between Articles and their corresponding explanatory Recitals.
- **High-Fidelity PDF Extraction**: Converts complex legal PDFs into structured Markdown, maintaining semantic hierarchy (Chapters, Articles, Sections).
- **Token-Optimized Chunking**: Uses `tiktoken` (GPT-4 logic) with sliding window overlap (500 tokens / 50 overlap) to preserve context across legal boundaries.
- **Advanced Ranking**: Implements **Reciprocal Rank Fusion (RRF)** to merge dense vector scores (all-MiniLM-L6-v2) with sparse keyword scores (BM25Okapi).
- **Expert Answer Generation**: Powered by `gemini-2.0-flash` for high-speed, grounded, and context-aware responses.

---

## 🏗️ Project Architecture

```text
.
├── app.py                       # Interactive Streamlit Chat Interface
├── main.py                      # Master Pipeline Orchestrator
├── services/                    # Domain-Specific Services
│   ├── scrap_mapping.py         # Web-scrapers for Article-to-Recital relationships
│   ├── extract_pdf.py           # Robust PDF to Markdown engine
│   ├── chunker.py               # Systematic GDPR structural parser
│   ├── vector_chunk.py          # Token-optimized sub-chunking service
│   └── rag_with_vector_db.py    # The Advanced Multi-Stage RAG Engine
├── output/                      # Data Artifacts
│   ├── scrap_mapping.json       # Synced legal mappings
│   ├── gdpr.md                  # Unified GDPR document base
│   ├── gdpr_articles.json       # Logically split legal units
│   ├── gdpr_recitals.json       # Extracted Recital database
│   └── gdpr_vector_chunks.json  # Search-ready vector segments
└── .env                         # Secure Environment Config
```

---

## 🚀 Execution Workflow

The system operates as a five-stage intelligent pipeline:

1.  **Sync Stage (`scrap_mapping.py`)**: Synchronizes Article-to-Recital mapping from the web to ensure legal cross-references are accurate.
2.  **Extraction Stage (`extract_pdf.py`)**: Parses the source PDF into a clean, hierarchical Markdown format.
3.  **Logical Chunking Stage (`chunker.py`)**: Splits the document into Chapters and Articles while embedding Recital cross-references.
4.  **Vectorization Stage (`vector_chunk.py`)**: Breaks down long articles into token-aligned segments for optimal embedding performance.
5.  **Intelligence Stage (`app.py` / `rag_with_vector_db.py`)**: The RAG interface where user queries trigger the multi-stage hybrid search and final answer generation.

---

## 🛠️ Installation & Setup

### 1. Environment Preparation
```bash
# Install required packages
pip install -r requirements.txt

# Configure API Access
cp .env.example .env
# Edit .env and insert your GEMINI_API_KEY
```

### 2. Run the Full Data Pipeline (Optional if data exists)
Run the orchestrator to build the unified knowledge base from the PDF:
```bash
python main.py
```

### 3. Launch the AI Interface
You can start the interactive chat interface using Streamlit:
```bash
streamlit run app.py
```

Or run the RAG engine directly for a CLI test:
```bash
python services/rag_with_vector_db.py
```

---

## 🧠 The RAG Engine & Python API

The core intelligence resides in `RagWithVectorDb`, which can be integrated directly into any Python application.

### API Usage Example

```python
from services.rag_with_vector_db import RagWithVectorDb

# Initialize and build indices
rag = RagWithVectorDb()
rag.build_index()

# Execute multi-stage query
# top_k: Number of article chunks to retrieve
# top_recitals: Number of supporting recitals to filter
response = rag.query(
    question="What are the rules for data breach notifications?", 
    top_k=3, 
    top_recitals=2
)

print(f"Expert Answer: {response['answer']}")
print(f"Sources: {len(response['retrieved_chunks'])} articles, {len(response['preferred_recitals'])} recitals")
```

| Component | Technology |
|---|---|
| **Semantic Search** | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| **Keyword Search** | **BM25Okapi** |
| **Vector DB** | FAISS `IndexFlatIP` (Inner Product / Cosine Similarity) |
| **Fusion Algorithm**| **Reciprocal Rank Fusion (RRF)** |
| **LLM Model** | `gemini-2.0-flash` |

### Multi-Stage Logic Diagram
1. **Query** ➔ **Hybrid Search (Articles)** ➔ **Top K Articles**
2. **Top K Articles** ➔ **Associated Recital Extraction** ➔ **Candidate Recital Pool**
3. **Query** + **Candidate Pool** ➔ **Hybrid Search (Recitals)** ➔ **Top Supporting Recitals**
4. **Final Filtered Articles** + **Top Recitals** ➔ **Gemini Expert** ➔ **Grounded Answer**

---

## 📊 Knowledge Structure (JSON Entry)

Final vector chunks preserve deep legal context:

```json
{
    "content": "Article 9: Processing of special categories of personal data...",
    "metadata": {
        "chapter_name": "Rights of the data subject",
        "article_name": "Right to erasure (‘right to be forgotten’)",
        "recitals": ["65", "66"],
        "token_count": 412,
        "part": "part 1"
    }
}
```

---

*Developed as a premium GDPR compliance intelligence tool.*

---

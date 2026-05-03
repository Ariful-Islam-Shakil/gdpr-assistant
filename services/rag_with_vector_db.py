import os
import json
import numpy as np
import faiss
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from rank_bm25 import BM25Okapi

load_dotenv()

class RagWithVectorDb:
    """
    RAG pipeline for GDPR compliance documentation using:
      - Hybrid Search (FAISS + BM25) for high-relevance retrieval.
      - Dual-Stage Retrieval: Articles first, then associated Recitals.
      - Gemini 2.5 Flash for accurate response generation.
    """

    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    # GEMINI_MODEL_NAME = "gemini-2.0-flash"
    GEMINI_MODEL_NAME = "gemini-3-flash-preview"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "output", "gdpr_vector_chunks.json")
    RECITAL_DATA_PATH = os.path.join(BASE_DIR, "output", "gdpr_recitals.json")

    def __init__(self, data_path: str = None):
        """
        Initializes models, indices, and connects to the Gemini API.
        """
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        
        # Initialize Embedding Model
        print(f"[RagWithVectorDb] Loading embedding model: {self.EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL_NAME)

        # Initialize Indices and Storage
        self.index: faiss.IndexFlatIP | None = None  # Semantic Index
        self.bm25: BM25Okapi | None = None          # Keyword Index
        self.chunk_store: list[dict] = []           # In-memory chunk storage

        # Initialize Gemini Client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in .env file.")
        self.llm = genai.Client(api_key=api_key)
        
        # Load Recital Mappings
        self.recital_map = self._load_recital_data()

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def build_index(self) -> None:
        """
        Loads the vector chunks and builds both Semantic (FAISS) and Sparse (BM25) indices.
        """
        self.chunk_store = self._load_json(self.data_path)
        
        # 1. Build Semantic Index (FAISS)
        contents = [f"Chapter: {c['metadata']['chapter']} - {c['metadata']['chapter_name']}\nArticle: {c['metadata']['article']} - {c['metadata']['article_name']}\n{c['content']}" for c in self.chunk_store]
        embeddings = self._embed_texts(contents)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        
        # 2. Build Keyword Index (BM25)
        tokenized_corpus = [self._tokenize(c) for c in contents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"[RagWithVectorDb] Indices built with {len(self.chunk_store)} chunks.")

    def query(self, question: str, hybrid: bool = True, top_k: int = 5, top_recitals: int = 3) -> str:
        """
            Executes the full RAG pipeline:
            1. Hybrid Search on Articles.
            2. Identify associated Recitals.
            3. Hybrid search on identified Recitals to filter top context.
            4. Generate final answer.
        """
        # 1. Initial Hybrid Search on Vector DB
        rewritten_query = self.query_rewrite(question)
        print(f"\n***********\nRewritten Query: {rewritten_query}")
        retrieved_chunks = self.rag_search(rewritten_query, top_k=top_k, hybrid=hybrid)
        
        # 2. Extract related Recitals from metadata
        candidate_recital_texts = self._extract_associated_recitals(retrieved_chunks)
        
        # 3. Hybrid Search on Recitals to find the most relevant ones
        preferred_recitals = self.hybrid_search_texts(rewritten_query, candidate_recital_texts, top_k=top_recitals, hybrid=hybrid)
        
        if "irrelevant query" in rewritten_query.lower():
            retrieved_chunks = []
            preferred_recitals = []
        # 4. Generate Final Answer
        answer = self.generate_response(question, retrieved_chunks, preferred_recitals)
        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "preferred_recitals": preferred_recitals
        }

    def rag_search(self, query: str, hybrid: bool = True, top_k: int = 5) -> list[dict]:
        """
        Performs semantic search or hybrid search (Semantic + Keyword).
        Uses Reciprocal Rank Fusion (RRF) if hybrid=True.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Please call build_index() first.")

        # Semantic Search
        query_vec = self._embed_texts([query])
        dense_scores, dense_indices = self.index.search(query_vec, top_k * 2)
        semantic_ranked = dense_indices[0].tolist()
        print(f"\n***********\nSemantic Ranked: {dense_indices}")
        print(f"\n***********\nSemantic Scores: {dense_scores}")

        # If hybrid is False → return only semantic results
        if not hybrid:
            top_indices = semantic_ranked[:top_k]
            return [self.chunk_store[idx] for idx in top_indices]

        # If hybrid=True → need BM25
        if self.bm25 is None:
            raise RuntimeError("BM25 not built. Please call build_index() first.")

        # Keyword Search
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        keyword_ranked = np.argsort(bm25_scores)[::-1][:top_k * 2].tolist()
        print(f"\n***********\nKeyword Ranked: {keyword_ranked}")
        print(f"\n***********\nKeyword Scores: {bm25_scores}")

        # Merge using RRF
        merged_indices = self._rrf(semantic_ranked, keyword_ranked, top_k)
        return [self.chunk_store[idx] for idx in merged_indices]

    def hybrid_search_texts(self, query: str, text_list: list[str], hybrid: bool = True, top_k: int = 3) -> list[str]:
        """
        Performs semantic search or hybrid search on a dynamic list of strings.
        If hybrid=False → semantic only.
        """
        rewritten_query = self.query_rewrite(query)
        if not text_list:
            return []

        # 1. Semantic Ranking
        embeddings = self._embed_texts(text_list)
        query_vec = self._embed_texts([query])
        semantic_scores = (embeddings @ query_vec.T).flatten()
        semantic_ranked = np.argsort(semantic_scores)[::-1].tolist()

        # If hybrid is False → return semantic only
        if not hybrid:
            top_indices = semantic_ranked[:top_k]
            return [text_list[i] for i in top_indices]

        # 2. Keyword Ranking (only if hybrid=True)
        tokenized_corpus = [self._tokenize(t) for t in text_list]
        bm25_temp = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25_temp.get_scores(self._tokenize(query))
        keyword_ranked = np.argsort(bm25_scores)[::-1].tolist()

        # 3. Merge with RRF
        top_indices = self._rrf(semantic_ranked, keyword_ranked, top_k)
        return [text_list[i] for i in top_indices]

    def generate_response(self, query: str, chunks: list[dict], recitals: list[str]) -> str:
        """
        Constructs a prompt and gets a response from Gemini.
        """
        context_block = self._format_context(chunks, recitals)
        prompt = self._build_prompt(query, context_block)

        response = self.llm.models.generate_content(
            model=self.GEMINI_MODEL_NAME,
            contents=prompt,
        )
        return response.text

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _load_json(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_recital_data(self) -> dict:
        try:
            recitals = self._load_json(self.RECITAL_DATA_PATH)
            return {str(r["number"]): r["content"] for r in recitals}
        except Exception as e:
            print(f"[Warning] Could not load recitals: {e}")
            return {}

    def _extract_associated_recitals(self, chunks: list[dict]) -> list[str]:
        """Collects unique recital contents based on metadata in retrieved chunks."""
        recital_numbers = []
        for chunk in chunks:
            nums = chunk.get("metadata", {}).get("recitals", [])
            recital_numbers.extend(nums)
        
        unique_nums = sorted(list(set(recital_numbers)))
        results = []
        for num in unique_nums:
            content = self.recital_map.get(str(num))
            if content:
                results.append(f"Recital ({num}): {content}")
        return results

    def _tokenize(self, text: str) -> list[str]:
        """Simple cleaning and tokenization."""
        return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).split()

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generates normalized embeddings for silver-standard cosine similarity."""
        return self.embedding_model.encode(
            texts, 
            normalize_embeddings=True, 
            show_progress_bar=False
        ).astype("float32")

    def _rrf(self, list1: list[int], list2: list[int], top_k: int, c: int = 60) -> list[int]:
        """Reciprocal Rank Fusion algorithm to combine two ranked lists."""
        scores = {}
        for rank, idx in enumerate(list1):
            if idx == -1: continue
            scores[idx] = scores.get(idx, 0) + 1.0 / (c + rank + 1)
        for rank, idx in enumerate(list2):
            if idx == -1: continue
            scores[idx] = scores.get(idx, 0) + 1.0 / (c + rank + 1)
        
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

    def _format_context(self, chunks: list[dict], recitals: list[str]) -> str:
        """Prepares a structured context block for the LLM."""
        sections = []
        
        # Article Sections
        if chunks:
            sections.append("### Relevant Articles")
            for i, chunk in enumerate(chunks, 1):
                meta = chunk.get("metadata", {})
                header = f"[{i}] Chapter: {meta.get('chapter', 'Unknown Chapter')} - {meta.get('chapter_name', 'Unknown Chapter')}\nArticle: {meta.get('article', 'Unknown Article')} - {meta.get('article_name', 'Unknown Article')}"
                sections.append(f"{header}\n{chunk['content']}")

        # Recital Sections
        if recitals:
            sections.append("\n### Supporting Recitals")
            sections.extend(recitals)
            
        return "\n\n".join(sections)

    def _build_prompt(self, query: str, context: str) -> str:
        # with open("context.txt", "w") as f:
        #     f.write(context)
        return (
            "You are a highly experienced GDPR legal expert.\n"
            "Answer the question using ONLY the provided context below.\n"
            "If the context does not contain the answer, state that clearly.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            "EXPERT ANSWER:"
        )
    def get_article_names(self)-> str:
        data = self._load_json(self.DEFAULT_DATA_PATH)
        article_names = [] 
        for chunk in data:
            chapter = f"{chunk['metadata']['chapter']} - {chunk['metadata']['chapter_name']}"
            article = f"{chunk['metadata']['article']} - {chunk['metadata']['article_name']}"
            if f"\n## {chapter}\n" not in article_names: 
                article_names.append(f"\n## {chapter}\n")
            article_names.append(f"  - ARTICLE: {article}")
        return "\n".join(article_names)
    def query_rewrite(self, query: str) -> str:
        """
        Rewrites the query to be more suitable for semantic search.
        """
        article_names = self.get_article_names()
        prompt = (
            "You are a legal query rewriting expert specializing in GDPR.\n"
            "Rewrite the user's question so that it is optimized for semantic search "
            "over official GDPR documents.\n\n"
            "Instructions:\n"
            "- Use formal legal terminology.\n"
            "- Align the question with relevant GDPR article titles or headings if applicable.\n"
            "- Include key legal concepts (e.g., lawfulness of processing, data minimisation, purpose limitation, legal basis).\n"
            "- Do NOT introduce new facts.\n"
            "- Keep the meaning exactly the same.\n\n"
            "- If the question is clearly unrelated to GDPR, personal data protection, or EU data protection law, return exactly: irrelevant query\n\n"
            f"AVAILABLE ARTICLE HEADERS:\n{article_names}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            "REWRITTEN LEGAL SEARCH QUERY:"
        )
        response = self.llm.models.generate_content(
            model=self.GEMINI_MODEL_NAME,
            contents=prompt,
        )
        return response.text

# ---------------------------------------------------------------------------
# Execution Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rag = RagWithVectorDb()
    rag.build_index()

    test_query = "What are the rules for processing sensitive data?"
    print(f"\n[Query]: {test_query}")
    
    response = rag.query(test_query, hybrid=False, top_k=3, top_recitals=3)
    print(f"\n[Answer]:\n{response['answer']}")

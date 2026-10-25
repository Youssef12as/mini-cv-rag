# CV-Based RAG Application

An intelligent Retrieval-Augmented Generation (RAG) assistant that uses Youssef Sayed's CV (Curriculum Vitae) as its knowledge base to answer technical recruiter and interviewer questions factually and without hallucinations.

The project demonstrates the evolution from a baseline **Naive RAG** implementation (Task 1) to an **Advanced RAG** pipeline (Task 2) incorporating targeted **Pre-Retrieval** and **Post-Retrieval** optimization techniques.

---

## Architecture Overview

```
[User Question]
       │
       ▼
═════════════════════════════════════════════════════════════════
1. PRE-RETRIEVAL (Task 2: Multi-Query Expansion)
   Generates 3 diverse query formulations capturing synonyms & technical terms
═════════════════════════════════════════════════════════════════
       │
       ▼  (Parallel Search on Chroma Vector Index)
═════════════════════════════════════════════════════════════════
2. RETRIEVAL & VECTOR STORE
   OpenAI text-embedding-3-small (1536 dims) over recursive chunks
═════════════════════════════════════════════════════════════════
       │
       ▼  (Candidate Pool: fetch_k = 10)
═════════════════════════════════════════════════════════════════
3. POST-RETRIEVAL (Task 2: Maximal Marginal Relevance - MMR)
   Balances relevance vs. novelty (lambda=0.5) to select 4 diverse chunks
═════════════════════════════════════════════════════════════════
       │
       ▼
═════════════════════════════════════════════════════════════════
4. GENERATION & FACTUAL GROUNDING
   Augmented Context + Strict Grounding Prompt -> GPT-4o-mini (temp=0)
   Output: Factual Answer OR Abstention ("This information is not available")
═════════════════════════════════════════════════════════════════
```

---

## Task 2: Pre-Retrieval & Post-Retrieval Implementation

Per Task 2 requirements, two advanced techniques were integrated into the pipeline with concrete justifications:

### 1. Pre-Retrieval Technique: Multi-Query Expansion
* **Implementation:** When a user submits a question, a lightweight prompt passes the query to `gpt-4o-mini` to generate **3 alternative query formulations** from different technical angles and synonyms (e.g., mapping broad terms like "backend" into "FastAPI, Flask, APIs"). Each query is executed against the vector database and the resulting candidate sets are unioned and deduplicated.
* **Why Chosen for CV QA (Justification):**
  - **Vocabulary Mismatch Problem:** Recruiters and interviewers frequently ask broad or conversational questions (*"What server-side technologies does he use?"*), whereas the CV text contains concrete, specific tool names (*"FastAPI, Node.js, REST API Design"*).
  - Single-query dense embeddings can miss relevant chunks if the user's phrasing is slightly semantically distant from the bullet points.
  - Multi-Query Expansion broadens the search surface area, significantly increasing **Retrieval Recall** without requiring manual prompt crafting from the user.

### 2. Post-Retrieval Technique: Maximal Marginal Relevance (MMR) Diversity Control
* **Implementation:** Instead of naive greedy top-K cosine similarity, retrieval uses the **MMR algorithm**:
  $$\text{MMR} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}_2(d_i, d_j) \right]$$
  The retriever fetches an initial candidate pool of $fetch\_k = 10$ chunks and selects $k = 4$ chunks using $\lambda = 0.5$.
* **Why Chosen for CV QA (Justification):**
  - **Context Cannibalization Problem:** In a compact 2-page CV, the largest section (Projects) contains multiple chunks with high dense similarity. Standard cosine search frequently returns 3 or 4 chunks all describing the same ML project, completely starving the context window of other critical sections.
  - **Cross-Sectional Coverage:** MMR penalizes redundant chunks that are too similar to already selected passages. This guarantees that the final prompt context contains a balanced representation across **Projects, Technical Skills, Education, and Competitive Programming (ECPC)**.

---

## Comparison: Baseline Naive RAG vs. Advanced RAG

Run the built-in comparison mode to see the direct impact:
```bash
python main.py --compare "What are Youssef's main projects and achievements?"
```

| Evaluation Metric | Baseline Naive RAG (Task 1) | Advanced RAG (Task 2) |
|---|---|---|
| **Query Mechanism** | Single query, raw user text | Multi-Query Expansion (3 diverse formulations) |
| **Search Algorithm** | Standard Cosine Similarity Top-4 | MMR ($\lambda=0.5$, $fetch\_k=10$, $k=4$) |
| **Coverage Quality** | Concentrated on Deepfake & Hakawi; **missed ECPC qualification** | **Retrieved ECPC Rank 23 out of 300+ teams** alongside Hakawi & Q.E.D. |
| **Redundancy** | High (multiple chunks repeating ML bullets) | Low (diverse chunks from Projects, Skills, and Leadership) |
| **Hallucination Safety** | Strict prompt abstention | Strict prompt abstention + diverse evidence |

---

## Project Structure

```
cv-rag-app/
├── data/
│   └── Youssef_Sayed-CV.pdf   # Primary CV document (PDF format)
├── main.py                    # Complete RAG application (Interactive, Single-Query & Compare modes)
├── requirements.txt           # Minimal dependencies
├── .env.example               # Template for environment variables (OpenAI API key)
├── .gitignore                 # Excludes .env and local cache from version control
└── README.md                  # Comprehensive documentation and Task 2 report
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Youssef12as/mini-cv-rag.git
cd mini-cv-rag
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv

# On Linux / macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your OpenAI API Key
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```
Add your OpenAI API key to `.env`:
```ini
OPENAI_API_KEY=sk-your-actual-api-key-here
```

---

## How to Run

### 1. Interactive Shell (Default: Advanced Mode)
```bash
python main.py
```
*Prompt:*
```
Ask a question: What backend frameworks does Youssef know?
Answer:
Youssef knows FastAPI, Flask, and Node.js.

Ask a question: Does Youssef know Rust?
Answer:
This information is not available in the CV.
```

### 2. Side-by-Side Comparison Mode (Naive vs. Advanced)
Compare Task 1 (Naive) against Task 2 (Advanced) on any question:
```bash
python main.py --compare "What are Youssef's main projects and achievements?"
```
Or inside the interactive shell:
```
Ask a question: --compare What are Youssef's main projects and achievements?
```

### 3. Single-Query Command Line Mode
```bash
python main.py "What university does Youssef attend and what is his grade?"
```

---

## Technology Stack

| Component | Technology | Role |
|---|---|---|
| Orchestration | LangChain (v0.2+) | Pipeline execution & LCEL chains |
| Document Parsing | PyPDF | Native PDF extraction from `Youssef_Sayed-CV.pdf` |
| Text Chunking | RecursiveCharacterTextSplitter | Chunk size 500 chars, 100 overlap |
| Embeddings | OpenAI `text-embedding-3-small` | 1536-dimensional dense vector embeddings |
| Vector Store | ChromaDB | In-memory embedding storage and MMR indexing |
| Pre-Retrieval | Multi-Query Expansion via LLM | Generates 3 complementary query formulations |
| Post-Retrieval | Maximal Marginal Relevance (MMR) | Diversity selection ($\lambda=0.5$, $fetch\_k=10$) |
| Generation LLM | OpenAI `gpt-4o-mini` | Deterministic generation (temperature = 0) |

---

## Author

**Youssef Sayed**  
Computer Engineering, Cairo University  
GitHub: [Youssef12as](https://github.com/Youssef12as)  
Email: youssefnafady492@gmail.com

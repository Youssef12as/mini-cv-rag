# CV-Based Naive RAG Application

A Retrieval-Augmented Generation (RAG) application that uses a personal CV (Curriculum Vitae) as its sole knowledge base to answer user queries factually.

Built according to strict **Naive RAG** specifications without advanced retrieval optimization techniques (no query rewriting, query expansion, HyDE, reranking, or contextual compression).

---

## Architecture Overview

The application follows the classic linear Naive RAG pipeline:

```
[CV Document] (PDF / Text)
       │
       ▼
[Text Splitting] (RecursiveCharacterTextSplitter: chunk_size=500, overlap=100)
       │
       ▼
[Vector Embedding] (OpenAI text-embedding-3-small, 1536 dims)
       │
       ▼
[Vector Store] (ChromaDB In-Memory)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
 [User Query]                  [Similarity Search]
       │                           (Top-4 Chunks)
       │                                 │
       └──────────────┬──────────────────┘
                      ▼
            [Prompt Augmentation]
            (Context + Grounding Constraints)
                      │
                      ▼
               [LLM Generation]
               (GPT-4o-mini, temp=0)
                      │
                      ▼
             [Grounded Response]
             (Answers or Declines)
```

---

## Key Features

1. **Strict Context Grounding:** The system prompt explicitly restricts the LLM to facts present in the retrieved chunks.
2. **Abstention Mechanism (Handling Missing Information):** If a question asks about details not present in the CV, the system refuses to answer and outputs `"This information is not available in the CV."` rather than hallucinating.
3. **Dual Interface Support:** Can be executed via an interactive Jupyter Notebook (`naive_rag_cv.ipynb`) or through a command-line interface (`main.py`).
4. **Automated Evaluation Suite:** 13 test cases covering in-scope, out-of-scope, and edge questions with latency and chunk diagnostics.

---

## Technology Stack

| Component | Technology | Version / Spec |
|---|---|---|
| Orchestration Framework | LangChain | >= 0.2.0 |
| Document Parser | PyPDF / PyMuPDF | PDF & TXT support |
| Text Chunking | RecursiveCharacterTextSplitter | 500 chars, 100 overlap |
| Embedding Model | OpenAI `text-embedding-3-small` | 1536 dimensions |
| Vector Store | ChromaDB | In-memory collection |
| Generation LLM | OpenAI `gpt-4o-mini` | temperature = 0 |
| Configuration | python-dotenv | Environment variable loading |

---

## Project Structure

```
cv-rag-app/
├── data/
│   ├── Youssef_Sayed-CV.pdf   # Primary CV document (PDF format)
│   └── youssef_cv.txt         # Plain text version for fallback
├── naive_rag_cv.ipynb         # Main Jupyter Notebook with step-by-step pipeline & evaluation
├── main.py                    # Standalone CLI interface (interactive and single-query)
├── requirements.txt           # Python package dependencies
├── .env.example               # Template for environment variables (OpenAI API key)
├── .gitignore                 # Prevents committing secrets (.env) and caches
└── README.md                  # Project documentation and reproduction instructions
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/cv-rag-app.git
cd cv-rag-app
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
Open `.env` and add your OpenAI API key:
```ini
OPENAI_API_KEY=sk-your-actual-api-key-here
```

> **Security Note:** Never commit your `.env` file to version control. The `.gitignore` is already configured to exclude it.

---

## How to Run

### Option A: Jupyter Notebook (Recommended for Evaluation)
Start Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook naive_rag_cv.ipynb
```
Run all cells sequentially:
- **Sections 1–4:** Load, chunk, embed, and index the CV.
- **Section 5:** Build the LCEL RAG chain.
- **Section 6–7:** Run the evaluation test suite across in-scope, out-of-scope, and edge questions.
- **Section 8:** View the tabulated evaluation summary with latencies.

### Option B: Terminal / CLI Interface
**Interactive Shell:**
```bash
python main.py
```
*Prompt:*
```
Ask a question: What university does Youssef attend?
Answer:
Youssef attends Cairo University, pursuing a Bachelor of Engineering in Computer Engineering.

Ask a question: Does Youssef know Rust?
Answer:
This information is not available in the CV.
```

**Single-Query Mode:**
```bash
python main.py "What are Youssef's main projects?"
```

---

## Evaluation & Test Scenarios

The system was evaluated against three distinct query categories:

### 1. In-Scope Questions (Answers Present in CV)
| Question | Expected Outcome |
|---|---|
| *"What university does Youssef attend and what is his grade?"* | Cairo University, Computer Engineering, Cumulative: Excellent |
| *"What programming languages does Youssef know?"* | Python, C/C++, JavaScript, SQL |
| *"Describe the Hakawi project."* | AI Cultural Heritage Platform, RAG, TTS voice cloning, Phoenix observability |
| *"What was Youssef's rank in ECPC qualifications?"* | Rank 23 out of 300+ teams |

### 2. Out-of-Scope Questions (Abstention Verification)
| Question | Expected Outcome | Actual Behavior |
|---|---|---|
| *"What is Youssef's expected salary?"* | Refusal | `"This information is not available in the CV."` |
| *"Does Youssef know Rust or Go?"* | Refusal | `"This information is not available in the CV."` |
| *"What is the capital of France?"* | Refusal | `"This information is not available in the CV."` |
| *"What was Youssef's Thanaweya Amma score?"* | Refusal | `"This information is not available in the CV."` |

### 3. Edge Cases (Broad / Analytical)
| Question | Expected Outcome |
|---|---|
| *"Give a brief summary of Youssef's profile."* | Synthesizes education, core ML/deepfake projects, and competitive programming achievements without hallucinating unlisted roles. |

---

## Scope Restriction Compliance

As instructed in the assignment specification:
- **No Query Rewriting:** User queries are passed directly to vector embedding without LLM rewriting.
- **No Query Expansion / Multi-Query:** Only a single dense retrieval query is executed per question.
- **No HyDE:** No synthetic hypothetical documents are generated prior to search.
- **No Reranking:** Top-4 candidates from the vector store are directly passed to the context window without cross-encoder scoring.
- **No Contextual Compression:** Chunks are retrieved and provided as intact passages.

---

## Author

**Youssef Sayed**  
Computer Engineering, Cairo University  
Email: youssefnafady492@gmail.com

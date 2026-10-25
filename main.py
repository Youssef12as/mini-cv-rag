import os
import sys
import warnings

# Suppress library warnings for clean CLI output
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()

PROMPT_TEMPLATE = """You are an assistant answering questions about Youssef Sayed's CV.
Use ONLY the provided context to answer the question.
If the context does not contain enough information to answer, state clearly:
"This information is not available in the CV."
Do not extrapolate, assume, or fabricate any facts not explicitly present in the context.

Context:
{context}

Question: {question}

Answer:"""


def load_cv_documents():
    pdf_path = os.path.join(BASE_DIR, "data", "Youssef_Sayed-CV.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"CV document not found at: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    return loader.load()


# =====================================================================
# TASK 2: PRE-RETRIEVAL TECHNIQUE -> Multi-Query Expansion
#
# WHY CHOSEN:
# Recruiter and interview questions often use generalized terminology
# (e.g., "backend frameworks" or "cybersecurity"), while the CV lists
# concrete names (e.g., "FastAPI, Flask, Node.js", "deepfake detection").
# Multi-query expansion generates 3 complementary search formulations,
# solving the "Vocabulary Mismatch" problem and maximizing retrieval recall.
# =====================================================================
def expand_query(query: str, llm: ChatOpenAI) -> list[str]:
    """Pre-retrieval: Expands user query into 3 diverse formulations."""
    expansion_prompt = ChatPromptTemplate.from_template(
        "You are an assistant for a CV retrieval system.\n"
        "Given the user query about a candidate, generate 3 diverse search queries that capture\n"
        "different keywords, technical synonyms, and perspectives relevant to a CV.\n"
        "Return ONLY the 3 queries, one per line, without numbering or bullets.\n\n"
        "User Query: {query}\n\n"
        "Search Queries:"
    )
    chain = expansion_prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({"query": query})
        queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
        if query not in queries:
            queries.insert(0, query)
        return queries[:3]
    except Exception:
        return [query]


def build_pipeline():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[!] Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file containing: OPENAI_API_KEY=your_key_here\n")
        sys.exit(1)

    documents = load_cv_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="cv_kb"
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # 1. Baseline Naive Retriever (Standard Cosine Similarity Top-4)
    naive_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # =====================================================================
    # TASK 2: POST-RETRIEVAL TECHNIQUE -> Maximal Marginal Relevance (MMR)
    #
    # WHY CHOSEN:
    # In a compact CV, standard top-K cosine similarity often retrieves
    # redundant chunks from the same dense section (e.g. 4 chunks from
    # machine learning projects), causing "Context Cannibalization" and
    # starving the context of other vital sections (Education, Awards, Skills).
    # MMR (lambda=0.5) balances relevance to query with novelty against
    # already selected chunks, ensuring diverse cross-sectional coverage.
    # =====================================================================
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # Baseline Naive Chain
    naive_chain = (
        {"context": naive_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {
        "llm": llm,
        "prompt": prompt,
        "format_docs": format_docs,
        "naive_retriever": naive_retriever,
        "mmr_retriever": mmr_retriever,
        "naive_chain": naive_chain
    }


def query_advanced(pipeline, question: str, verbose: bool = False):
    """Executes Advanced RAG: Pre-retrieval Multi-Query + Post-retrieval MMR."""
    llm = pipeline["llm"]
    mmr_retriever = pipeline["mmr_retriever"]
    prompt = pipeline["prompt"]
    format_docs = pipeline["format_docs"]

    # Step 1: Pre-retrieval Multi-Query Expansion
    expanded_queries = expand_query(question, llm)
    if verbose:
        print("\n[Pre-Retrieval: Multi-Query Expansion]")
        for i, q in enumerate(expanded_queries, 1):
            print(f"  {i}. {q}")

    # Step 2: Post-retrieval MMR Search with Deduplication
    seen_texts = set()
    retrieved_docs = []
    for q in expanded_queries:
        docs = mmr_retriever.invoke(q)
        for doc in docs:
            # Deduplicate by first 100 characters to catch slight overlap
            fingerprint = doc.page_content[:100].strip()
            if fingerprint not in seen_texts:
                seen_texts.add(fingerprint)
                retrieved_docs.append(doc)

    # Select top 4 diverse chunks
    final_docs = retrieved_docs[:4]

    if verbose:
        print("\n[Post-Retrieval: MMR Diversity Selection]")
        for i, d in enumerate(final_docs, 1):
            snippet = d.page_content[:90].replace("\n", " ")
            print(f"  Chunk {i}: {snippet}...")

    # Step 3: Generation
    context_str = format_docs(final_docs)
    gen_chain = prompt | llm | StrOutputParser()
    answer = gen_chain.invoke({"context": context_str, "question": question})

    return {
        "answer": answer,
        "expanded_queries": expanded_queries,
        "chunks": final_docs
    }


def compare_pipelines(pipeline, question: str):
    """Compares Baseline Naive RAG vs. Advanced RAG (Task 2)."""
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    # 1. Run Baseline Naive RAG
    naive_docs = pipeline["naive_retriever"].invoke(question)
    naive_answer = pipeline["naive_chain"].invoke(question)

    print("\n[1] BASELINE NAIVE RAG (Top-4 Cosine Similarity):")
    print("-" * 50)
    for i, d in enumerate(naive_docs, 1):
        print(f"  Chunk {i}: {d.page_content[:90].replace(chr(10), ' ')}...")
    print("\nNaive Answer:")
    print(naive_answer)

    # 2. Run Advanced RAG (Task 2)
    advanced_res = query_advanced(pipeline, question, verbose=True)

    print("\nAdvanced Answer (Pre-retrieval Expansion + Post-retrieval MMR):")
    print(advanced_res["answer"])
    print("=" * 70 + "\n")


def interactive_loop():
    print("=" * 65)
    print("CV-Based RAG Application - Youssef Sayed")
    print("Mode: Advanced RAG (Multi-Query Expansion + MMR Diversity)")
    print("Initializing pipeline...")
    print("=" * 65)

    pipeline = build_pipeline()
    print("Pipeline ready. Type your question (or 'exit' / 'q' to quit):")
    print("Tip: Type '--compare <question>' to see Naive vs Advanced side-by-side.\n")

    while True:
        try:
            query = input("Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break

            if query.startswith("--compare "):
                actual_q = query.replace("--compare ", "").strip()
                compare_pipelines(pipeline, actual_q)
            else:
                result = query_advanced(pipeline, query, verbose=True)
                print(f"\nAnswer:\n{result['answer']}\n")
                print("-" * 65)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def single_query(question: str):
    pipeline = build_pipeline()
    if question.startswith("--compare "):
        compare_pipelines(pipeline, question.replace("--compare ", "").strip())
    else:
        res = query_advanced(pipeline, question, verbose=False)
        print(res["answer"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_arg = " ".join(sys.argv[1:])
        single_query(query_arg)
    else:
        interactive_loop()

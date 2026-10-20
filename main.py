import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
    pdf_path = os.path.join("data", "Youssef_Sayed-CV.pdf")
    txt_path = os.path.join("data", "youssef_cv.txt")

    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
    elif os.path.exists(txt_path):
        loader = TextLoader(txt_path, encoding="utf-8")
    else:
        raise FileNotFoundError(f"CV document not found in data/ folder ({pdf_path} or {txt_path})")

    return loader.load()


def build_naive_rag_chain():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file containing: OPENAI_API_KEY=your_key_here")
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

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def interactive_loop():
    print("=" * 65)
    print("CV-Based Naive RAG Application - Youssef Sayed")
    print("Initializing vector store and pipeline...")
    print("=" * 65)

    chain, retriever = build_naive_rag_chain()
    print("Pipeline ready. Type your question (or 'exit' / 'q' to quit):\n")

    while True:
        try:
            query = input("Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break

            response = chain.invoke(query)
            print(f"\nAnswer:\n{response}\n")
            print("-" * 65)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def single_query(question: str):
    chain, _ = build_naive_rag_chain()
    response = chain.invoke(question)
    print(response)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_arg = " ".join(sys.argv[1:])
        single_query(query_arg)
    else:
        interactive_loop()

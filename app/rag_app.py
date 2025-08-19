import os
from hashlib import sha256

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def read_files():
    documents = []
    for src in ["pg236.txt", "pg1513.txt"]:
        with open(src, "r") as file:
            text = file.read()
        text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        documents.extend(
            [
                Document(
                    page_content=doc,
                    metadata={
                        "source": src,
                        "id": sha256(doc.encode("utf-8")).hexdigest(),
                    },
                )
                for doc in chunks
            ]
        )
    return documents


documents = read_files()


def format_from_source(docs):
    return "\n\n".join(
        [
            f"[Source:{doc.metadata.get('source', '--N/A--')}]{doc.page_content}"
            for doc in docs
        ]
    )


# 2
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# 2.5
# db = FAISS.from_texts(texts, embeddings)
# db.save_local("vector_store")

# 3
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 7})
# results = retriever.get_relevant_documents("What is AI?")

# 4
llm = LlamaCpp(
    model_path=f"{os.path.expanduser('~')}/Downloads/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=50,
    n_threads=12,
    verbose=False,
)

# 5
prompt = PromptTemplate(
    input_variables=["ctx", "qry"],
    template="Use the following context to answer the question.\n\nContext: {ctx}\n\nQuestion: {qry}\n\nAnswer:",
)

#  6
rag_chain = (
    {"ctx": retriever | format_from_source, "qry": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# print(rag_chain)
# query = "What are Avinash's hobbies?"

while True:
    try:
        print("=" * 50)
        query = input("Enter a prompt: ").strip()
        # query = 'difference between jungle book and romeo julet'
        if not query:
            print("Please enter a valid prompt.")
            continue
        elif query.lower() == "exit":
            print("Exiting...")
            break
        result = rag_chain.invoke(query)
        source_docs = retriever.invoke(query)
        for doc in source_docs:
            print(f"Source: {doc.metadata.get('source', '--N/A--')}")
        print(result)
        break
    except Exception as e:
        print(f"Error: {e}. Retrying...")

# while True:
#     try:
#         print('=' * 50)
#         query = input("Enter a prompt: ").strip()
#         if not query:
#             print("Please enter a valid prompt.")
#             continue
#         elif query.lower() == "exit":
#             print("Exiting...")
#             break
#         resp = llm(
#             query,
#             max_tokens=512
#         )
#         out = resp.get("choices", [{}])[0]["text"]
#         print(out)
#     except Exception as e:
#         print(f"Error: {e}. Retrying...")

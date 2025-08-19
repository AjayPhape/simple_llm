import json
import logging
import os
from hashlib import sha256
from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from psycopg2.extras import execute_values
from simple_llm.app.pg_db import DatabaseConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


embedding = get_embeddings()


def read_files() -> None:
    data_folder = Path("../data")

    for src in data_folder.glob("*.txt"):
        with src.open("r") as file:
            text = file.read()

        text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        params = []

        for doc in chunks:
            params.append(
                (
                    json.dumps(
                        {"source": src, "hash": sha256(doc.encode("utf-8")).hexdigest()}
                    ),
                    embedding.embed_query(
                        doc
                    ),  # Assuming embeddings is defined globally
                    doc,
                )
            )

        with DatabaseConnection() as conn:
            qry = 'INSERT INTO public.book (metadata, embedding, "content") VALUES %s'
            with conn.connection.cursor() as cr:
                execute_values(cr, qry, params)
                conn.connection.commit()


def build_inputs(q: str) -> dict:
    with DatabaseConnection() as conn:
        qry = """
            WITH tmp AS (
                SELECT
                    content,
                    metadata
                FROM
                    book
                ORDER BY
                    embedding <-> %(query_embedding)s::vector
                LIMIT %(k)s
            )
            SELECT 
                string_agg(content, '\\n\\n') AS aggregated_content,
                jsonb_agg(metadata) AS aggregated_metadata
            FROM
                tmp;
        """
        params = {"k": 6, "query_embedding": embedding.embed_query(q)}
        ctx, docs = conn.fetchone(qry, params)
    return {"qry": q, "ctx": ctx.strip(), "docs": docs}


# 4
llm = LlamaCpp(
    model_path=f"{os.path.expanduser('~')}/Downloads/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2300,
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
# rag_chain = (
#     {"ctx": lambda q: format_from_source(retriever.invoke(q)), "qry": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

rag_source_chain = RunnableLambda(build_inputs) | {
    "answer": (
        (lambda x: {"qry": x["qry"], "ctx": x["ctx"]})
        | prompt
        | llm
        | StrOutputParser()
    ),
    "source": lambda x: x["docs"],
}

while True:
    try:
        logger.info("=" * 50)
        query = input("Enter a prompt: ").strip()
        if not query:
            logger.info("Please enter a valid prompt.")
            continue
        elif query.lower() == "exit":
            logger.info("Exiting...")
            break
        result = rag_source_chain.invoke(query)
        logger.info(result)
    except Exception as e:
        logger.exception(f"Error: {e}. Retrying...")

import json
import logging
import os
from hashlib import sha256
from pathlib import Path

from fastapi import FastAPI, HTTPException
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaEmbeddings
from psycopg2.extras import execute_values
from pydantic import BaseModel

from simple_llm.app.pg_db import DatabaseConnection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def get_embeddings():
    # from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    # return HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     model_kwargs={"device": "cpu"},
    # )
    return OllamaEmbeddings(model="nomic-embed-text")


embedding = get_embeddings()


def read_files() -> None:
    data_folder = Path("../data")

    for src in data_folder.glob("*.txt"):
        with src.open("r") as file:
            text = file.read()

        text_splitter = CharacterTextSplitter(
            chunk_size=1024, chunk_overlap=200, separator="\n"
        )
        chunks = text_splitter.split_text(text)
        emb_list = embedding.embed_documents(chunks)
        params = []

        for idx, doc in enumerate(chunks):
            params.append(
                (
                    json.dumps(
                        {
                            "source": str(src),
                            "hash": sha256(doc.encode("utf-8")).hexdigest(),
                        }
                    ),
                    emb_list[idx],
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

app = FastAPI()


class QueryRequest(BaseModel):
    prompt: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        # Validate the input
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

        # Invoke the RAG chain
        result = rag_source_chain.invoke(request.prompt)

        # Prepare the response
        logger.info(f"Result {result}")
        response = QueryResponse(
            answer=result.get("answer").strip(),
            sources=result.get("source", []),
        )
        return response

    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

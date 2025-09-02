import logging
import os

from fastapi import FastAPI, HTTPException
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaEmbeddings
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

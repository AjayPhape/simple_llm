import gc
import gzip
import json
import logging
import os
import pickle
from hashlib import sha256
from pathlib import Path

from zstandard import ZstdCompressor, ZstdDecompressor
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

LLM_STATE_FILE = "../pkl/llm_state.bin"


def read_files() -> str:
    data_folder = Path("../data")
    text = ""
    for src in data_folder.glob("av135.txt"):
        with src.open("r") as file:
            text += file.read()
    return text.replace("\r\n", "\n")


# 4
llm = LlamaCpp(
    model_path=f"{os.path.expanduser('~')}/Downloads/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=4000,
    n_gpu_layers=48,
    n_threads=12,
    verbose=False,
    path_session='sessions/knowledge_session.session',
)


def save_knowledge(model):
    logger.info(f"Saving knowledge to LLM")
    system_msg = (
        "You are a precise assistant answering from the preloaded knowledge pack."
        " If the answer is missing, reply 'UNKNOWN'."
        " Be clear and concise in your answers."
    )
    user_msg = f"BEGIN_KNOWLEDGE\n\n{read_files()}\n\nEND_KNOWLEDGE"

    # msg = '\n\n'.join(['<|system|>', system_msg, '<|user|>', user_msg, '<|assistant|>'])
    msg = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    logger.info(msg)
    out = model.client.create_chat_completion(msg, temperature=0.2, max_tokens=500)
    logger.info(out)

    logger.info("LLM state saved to knowledge file.")
    # return model


def load_state(model):
    if not Path(LLM_STATE_FILE).exists():
        logger.info(f"LLM state file {LLM_STATE_FILE} does not exist. Saving state.")
        save_knowledge(model)
    else:
        model.load_state(Path(LLM_STATE_FILE).read_bytes())
        logger.info(f"LLM state loaded from {LLM_STATE_FILE}")
    return model


save_knowledge(llm)
# loaded_model = load_state(llm)

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
        sys_msg = "Answer only from preloaded knowledge. If missing, reply 'UNKNOWN'."
        # message = '\n'.join(['<|system|>', sys_msg, '<|user|>', query, '<|assistant|>'])
        msg = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": query},
        ]
        result = llm.invoke(msg, temperature=0.2, max_tokens=300)
        # result = llm.invoke(message)
        logger.info(result)
    except Exception as e:
        logger.exception(f"Error: {e}. Retrying...")

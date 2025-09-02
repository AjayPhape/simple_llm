import json
from hashlib import sha256
from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter
from psycopg2.extras import execute_values

from simple_llm.app.config import embedding
from simple_llm.app.pg_db import DatabaseConnection


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


read_files()

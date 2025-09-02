from langchain_ollama import OllamaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# from transformers import AutoModel, AutoTokenizer


# model_id = "sentence-transformers/all-MiniLM-L6-v2"
# AutoModel.from_pretrained(model_id).save_pretrained("./local_emb_model")
# AutoTokenizer.from_pretrained(model_id).save_pretrained("./local_emb_model")
def get_embeddings():
    # HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     model_kwargs={"device": "cpu"},
    # )
    return OllamaEmbeddings(model="nomic-embed-text")


emb = get_embeddings()
print(len(emb.embed_query("Hello world")))

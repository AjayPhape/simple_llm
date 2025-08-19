import os

from llama_cpp import Llama

llm = Llama(
    model_path=f"{os.path.expanduser('~')}/Downloads/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=1024,
    n_gpu_layers=48,
    n_threads=10,
    verbose=False,
)

while True:
    try:
        print("=" * 50)
        query = input("Enter a prompt: ").strip()
        if not query:
            print("Please enter a valid prompt.")
            continue
        elif query.lower() == "exit":
            print("Exiting...")
            break
        resp = llm(query, max_tokens=512)
        out = resp.get("choices", [{}])[0]["text"]
        print(out)
    except Exception as e:
        print(f"Error: {e}. Retrying...")

from huggingface_hub import snapshot_download
for repo, pat, dest in [
    ("Qwen/Qwen2.5-7B-Instruct-GGUF", ["qwen2.5-7b-instruct-q4_k_m-0000*-of-00002.gguf"], "qwen2.5-7b-instruct-gguf"),
    ("d3LLM/d3LLM_LLaDA", None, "d3LLM_LLaDA"),
]:
    d = snapshot_download(repo, allow_patterns=pat, local_dir=f"/mnt/nvme/models/d3llm/{dest}")
    print("ok", repo, d, flush=True)
print("== downloads done", flush=True)

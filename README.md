# d3LLM on a Tesla V100: does the diffusion-LLM speedup survive against llama.cpp Q4?

2026-10-25. One Tesla V100-PCIE-32GB (gateway), SM clock locked at 1380 MHz (`tools/v100`), so the numbers are
about 10% below an unlocked boost. Idea and plan: Grok, `experiments/ideas/2026-10-25-d3llm-on-v100.md`. Review:
experiments#3. Run by Claude.

**Question.** On the V100 at batch 1, does d3LLM-LLaDA (hao-ai-lab/d3LLM @ e5bfcac; weights `d3LLM/d3LLM_LLaDA`,
Apache-2.0) work in fp16 and stay close to its published GSM8K score? Is it faster than the strongest practical
autoregressive baseline, Qwen2.5-7B-Instruct Q4_K_M in llama.cpp?

## Setup
- **d3LLM:** its own `eval_llada.py`, in a local copy `eval_llada_v100.py` (`eval_llada_v100.patch`) with two
  changes. It loads in `D3_DTYPE` (default fp16) instead of bf16, and it loads straight onto the GPU
  (`device_map`). Without the second change, a single-process run loads all 16 GB into host RAM first, and the
  gateway has 16 GB. flash-attn is not installed, so the model code falls back to PyTorch SDPA.
  - Settings are the paper's GSM8K ones from `eval_scripts/llada_gsm8k_cot.sh`: `gen_length=256, steps=256,
    block_length=32, threshold=0.5, early_stop=True`, with `generate_multi_block_kv_cache` (`cache_delay_iter=2,
    refresh_interval=3`) or `generate_multi_block`.
  - Speed is d3LLM's own `show_speed`: useful tokens divided by wall time per request, including prompt reading,
    with the first 10 requests skipped.
- **Qwen:** official `Qwen/Qwen2.5-7B-Instruct-GGUF` Q4_K_M (4.68 GB) in llama.cpp 5e03bdd, sm_70 build.
  `llama-server -ngl 99 -c 4096 -np 1`, driven by the same lm_eval task through `local-chat-completions` with its
  chat template, greedy decoding. Speed comes from the server's prompt and generation timings for requests 11-100,
  the same skip as d3LLM.
- **Task:** `gsm8k_cot_zeroshot`, first 100 test problems (`--limit 100`), flexible-extract accuracy (±4.4 points
  at 95%). Both models answer through their chat template. Strict-match is 0 for every arm, because no model
  writes "The answer is".

## Results

| arm | GSM8K (100) | tok/s | s per question (11-100) | tokens per forward | VRAM |
| --- | --: | --: | --: | --: | --: |
| **d3LLM-LLaDA fp16, multi-block + KV cache** | **74%** | **85.4** | **2.71** | 8.98 | ~14 GB |
| d3LLM-LLaDA fp16, multi-block (no cache) | 73% | 76.9 | 3.01 | 8.82 | ~14 GB |
| d3LLM-LLaDA bf16, multi-block + KV cache | 72% | 16.3 | 14.28 | 9.07 | ~14 GB |
| **Qwen2.5-7B Q4_K_M, llama.cpp, 512-token budget** | **75%** | **107.0** | **2.48** | 1 | 4.4 GB model |
| Qwen2.5-7B Q4_K_M, llama.cpp, 256-token budget (paper's budget) | 49% | 106.6 | 2.14 | 1 | 4.4 GB model |
| *Paper: d3LLM-LLaDA on an A100 / H100* | *73.1%* | *183.3 / 288.9* | | *9.11* | |

`llama-bench` for Qwen Q4_K_M: pp512 2220 tok/s, tg128 111.2 tok/s.

## Findings
1. **d3LLM works in fp16 on Volta, and fp16 costs no accuracy.**
   - fp16 scores 74% and bf16 72% on the same card and questions, both against the paper's 73.1. The difference is
     inside the noise.
   - Tokens per forward pass are 8.98 against the paper's 9.11, so the parallel decoding behaves as published.
   - The 5-question smoke test produced coherent step-by-step answers with no NaN.
   - bf16 on the V100 is 5.2x slower (16.3 against 85.4 tok/s): there are no bf16 tensor cores. Use fp16.
2. **Against llama.cpp Q4 it's nearly a tie, and llama.cpp wins narrowly.** With Qwen given room to finish
   (512 tokens), accuracy is equal (75% against 74%). Qwen takes about 9% less time per question (2.48 s against
   2.71 s) and uses about a third of the memory (4.4 GB of weights against ~14 GB). Its tok/s is 25% higher, but it
   also writes more: 266 tokens per question against d3LLM's 231.
   - d3LLM's 3.6-5x headline comes from comparing against Hugging Face `generate` on newer GPUs. Against a real
     engine on bandwidth-limited old hardware, the advantage disappears.
   - It still turns the 274 ms-per-step diffusion LM we measured in September (volta-diffusion, Dream-7B Q8,
     ~7 tok/s) into an 85 tok/s one, within 10% of the best autoregressive engine.
3. **The paper's 256-token budget penalises wordy models.** At 256 tokens, 52 of Qwen's 100 answers were cut off
   before the final number, and it scored 49%. At 512 tokens, 2 were cut off and it scored 75%. d3LLM is distilled
   to answer briefly and finishes inside 256. Any comparison at the paper's budget measures length, not maths.
4. **The KV-cache mode is worth about 11%** (85.4 against 76.9 tok/s), at the same accuracy.

## Caveats
- n=100, so ±4.4 points. Only the big differences are readable: the bf16 slowdown, the 256-token collapse, and
  speed.
- The two speed figures are measured differently:
  - d3LLM's is Python wall time around each `generate`.
  - Qwen's comes from server-side timings, which exclude HTTP and lm_eval overhead, so they slightly favour Qwen.
  - The 9% gap is in that range, so call it a tie on speed.
- Qwen is 4-bit and d3LLM is 16-bit. No 4-bit d3LLM exists.
- Only GSM8K. d3LLM's distillation data is maths and code, so its acceleration on other prompts may be lower (its
  own chat script warns about this).
- d3LLM's VRAM figure is its own per-step "GPU Memory Used" print; Qwen's is the GGUF size only.

## Files
- `setup.sh`: venv copied from an existing torch 2.10 cu128 venv, then transformers 4.49.0, datasets 3.2.0,
  accelerate 1.10.1, the bundled lm-eval 0.4.9.1 with `[api]`, and einops 0.8.1.
- `patch_eval.py` / `eval_llada_v100.patch`
- `run_d3llm.sh`, `run_qwen.sh`, `run_qwen2.sh` (`MAXTOK`), `run_set.sh`
- `results/` holds the logs, lm_eval JSON and per-question samples.

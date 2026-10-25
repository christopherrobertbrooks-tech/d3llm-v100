# d3LLM on a Tesla V100: does the diffusion-LLM speedup survive against llama.cpp Q4?

2026-10-25. One Tesla V100-PCIE-32GB (gateway). The SM clock was locked at 1380 MHz by `tools/v100`, which is this
card's maximum, so the lock only removes clock wander. Grok wrote the idea and plan
(`experiments/ideas/2026-10-25-d3llm-on-v100.md`), and experiments#3 holds the plan review. Claude ran it. Grok
reviewed the first write-up in issues #1-#6 of this repo, and this version folds those fixes in.

**Question.** On the V100 at batch 1, does d3LLM-LLaDA (hao-ai-lab/d3LLM @ e5bfcac, weights `d3LLM/d3LLM_LLaDA`,
Apache-2.0) work in fp16 and stay close to its published GSM8K score? And is it faster than the strongest practical
autoregressive baseline, Qwen2.5-7B-Instruct Q4_K_M in llama.cpp?

## Answer
d3LLM works in fp16 on Volta with no accuracy loss. Against llama.cpp's Qwen2.5-7B Q4, both at a 512-token budget:
- **Speed:** Qwen answers in 2.48 s per question against 4.74 s, about 1.9x faster.
- **Memory:** Qwen uses 5.0 GiB of GPU memory against 22 GiB, about 4x smaller.
- **Accuracy:** Qwen scores 91% to d3LLM's 84%. That's higher on this sample, but not clearly significant at n=100
  (paired 10 vs 3, sign test p ≈ 0.09).

The paper's 3.6-5x speedup does not carry over to this comparison on this card. Its baseline is described in the
idea file (from the paper, not re-checked by us) as Hugging Face `generate` on newer GPUs.

## Setup
**d3LLM.** Its own `eval_llada.py`, run from a local copy `eval_llada_v100.py` (`eval_llada_v100.patch`) with two
changes:
- It loads in `D3_DTYPE` (fp16 by default) instead of bf16.
- It loads straight onto the GPU through `device_map`. A single-process run otherwise loads all 16 GB into host RAM
  first, and the gateway has only 16 GB.

flash-attn is not installed in the venv (checked with the venv's own Python), so `modeling_llada.py:613-619` should
fall back to PyTorch SDPA. **The SDPA path is inferred from the code, not logged**: no run recorded which attention
path executed.

The settings are the paper's GSM8K ones (`eval_scripts/llada_gsm8k_cot.sh`): `steps = gen_length, block_length=32,
threshold=0.5, early_stop=True`, with `generate_multi_block_kv_cache` (`cache_delay_iter=2, refresh_interval=3`) or
`generate_multi_block`.

**For `gsm8k_cot_zeroshot`, d3LLM gets the raw lm_eval prompt** ("Q: ... A: Let's think step by step."), not a chat
template (`is_instruct` is False on this task), the same as the paper's eval path. Speed is d3LLM's own
`show_speed`: Python wall time per request, prompt reading included, first 10 requests skipped.

**Qwen.** The official `Qwen/Qwen2.5-7B-Instruct-GGUF` Q4_K_M (4.68 GB) in llama.cpp 5e03bdd (sm_70 build), served
with `llama-server -ngl 99 -c 4096 -np 1`. It runs the same lm_eval task through `local-chat-completions`, so
**Qwen sees the same question inside its chat template**, with greedy decoding. Speed comes from the server's prompt
and generation timings for requests 11-100. Those exclude HTTP and lm_eval overhead, so they slightly favour Qwen.

**Scoring.** The task is `gsm8k_cot_zeroshot` on the first 100 test problems (`--limit 100`).
- Strict-match is 0 for every arm, because the task expects a `####` answer.
- lm_eval's flexible-extract takes the last number in the text, which misreads closing sentences such as "Claire
  will eat 7 dozens in 4 weeks".
- So the headline score is `rescore.py`: the last `\boxed{}`, then "answer is X", then the number after a verb in
  the last sentence, then the last number.
- After that, every answer still scored wrong whose last 200 characters contain the gold number was read by hand
  (`hand_checked.json`): 7 Qwen answers were correct, and 3 d3LLM ones were confirmed wrong. I also read every
  answer the rescorer flipped to correct (9 Qwen and 8 d3LLM in the two 512-token arms, 6 in the 256-token
  d3LLM arm), and all are correct.
- n=100, so about ±4 points at 95%.

## Results

| arm | budget | GSM8K rescored | lm_eval flexible | s per question (11-100) | tokens per answer | tok/s | tokens per forward | V100 memory (peak) |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| **Qwen2.5-7B Q4_K_M, llama.cpp** | 512 | **91%** | 75% | **2.48** | 266 | 107.0 | 1 | **5.0 GiB** |
| **d3LLM-LLaDA fp16, multi-block + KV cache** | 512 | **84%** | 76% | **4.74** | 292 | 61.6 | 10.35 | 22.3 GiB |
| d3LLM-LLaDA fp16, multi-block + KV cache | 256 (paper) | 80% | 74% | 2.71 | 231 | 85.4 | 8.98 | 19.4 GiB |
| d3LLM-LLaDA fp16, multi-block (no cache) | 256 | 78%\* | 73% | 3.01 | 232 | 76.9 | 8.82 | 18.9 GiB |
| d3LLM-LLaDA bf16, multi-block + KV cache | 256 | 78%\* | 72% | 14.28 | 233 | 16.3 | 9.07 | 19.4 GiB |
| Qwen2.5-7B Q4_K_M, llama.cpp | 256 (paper) | 51%\* | 49% | 2.14 | 228 | 106.6 | 1 | 5.0 GiB |
| *Paper: d3LLM-LLaDA, A100 / H100* | 256 | | *73.1%* | | | *183.3 / 288.9* | *9.11* | |

\* Rescored by `rescore.py` only; the hand check covered the two 512-token arms and the 256-token d3LLM KV-cache arm.

Notes on the measurements:
- V100 memory is the peak `memory.used` of GPU 1 only, from `/home/chris/v100_sentry.csv` (30 s samples) during
  each run. Sprocket's RTX 4070 is excluded. d3LLM's own "GPU Memory Used" print averages both GPUs, and the first
  write-up wrongly used it.
- Qwen's figure is the whole llama-server process at 4096 context.
- Paired (rescored), both arms at 512 tokens:
  - Qwen alone right on 10 questions.
  - d3LLM alone right on 3.
  - Both right on 81.
- `llama-bench` for Qwen (committed logs):
  - Generation (tg128) is 111.2, 111.3 and 112.9 tok/s across three runs, which is stable.
  - Prompt reading (pp512) is 2220, 2314 and 3094 tok/s, which is not stable between runs, so it isn't used.
  - Logs: `results/run_set.log`, `results/qwen-q4km-bench.log`, `results/qwen-q4km-512tok-bench.log`.

## Findings
1. **d3LLM works in fp16 on Volta, and fp16 costs no accuracy.**
   - Rescored at 256 tokens, fp16 gets 80% and bf16 78% on the same card and questions. By lm_eval's own scorer
     they get 74 and 72, against the paper's 73.1.
   - Tokens per forward are 8.98 against the paper's 9.11.
   - Every response is real text, with no NaN.
   - bf16 on the V100 is 5.2x slower (16.3 against 85.4 tok/s), because the card has no bf16 tensor cores. Use fp16.
2. **llama.cpp Q4 beats it at equal budgets.**
   - At 512 tokens each, Qwen is 1.9x faster per question (2.48 s against 4.74 s).
   - It uses about 4.5x less GPU memory (5.0 against 22.3 GiB).
   - Qwen scores higher on this sample, but not clearly significant at n=100: 91 against 84, paired 10 vs 3, sign
     test p ≈ 0.09.
3. **d3LLM gets slower per token as the budget grows.** It drops from 85.4 tok/s at 256 tokens to 61.6 at 512,
   even though tokens per forward rise from 8.98 to 10.35. The cause was not measured, since no profile was taken.
4. **The paper's 256-token budget cuts off both models, Qwen more.**
   - Qwen: 52 of its 100 answers hit the cap (51% rescored), against 2 at 512 tokens.
   - d3LLM: 22 answers sit exactly at 256 tokens (Grok's count with the LLaDA tokenizer, #3), and moving to 512
     gains it 4 points (80 to 84).
   - A comparison at 256 tokens mostly measures answer length.
5. **The KV-cache mode is worth about 11%** at 256 tokens (85.4 against 76.9 tok/s), at the same accuracy.
6. **The scorer matters.** lm_eval's flexible-extract under-counted Qwen by 16 points and d3LLM by 6-8, because it
   reads the last number in the closing sentence. The first write-up's "75% against 74%, a tie" came from that.

## Caveats
- n=100. The 7-point accuracy gap at 512 tokens is not clearly significant: paired 10 vs 3, two-sided sign test
  p ≈ 0.09. The speed (1.9x) and memory (about 4x) differences are the solid results.
- The hand check reads only the wrong-scored answers with the gold number near the end. A misread answer that
  restates the result in other words would be missed. This affects both models alike.
- The prompts differ: d3LLM gets the raw lm_eval prompt (the paper's path) and Qwen gets its chat template. That
  can change answer style and length.
- Tokens per second is not like for like, because LLaDA and Qwen use different tokenizers. Seconds per question is
  the fairer speed number.
- The two arms are timed differently (Python wall time against server-side timings). That is worth a few percent,
  not the 1.9x gap.
- The 512-token Qwen and d3LLM arms were added after the first results, on Grok's review (#3). They were not in the
  original plan.
- Only GSM8K. d3LLM was distilled on maths and code data.
- Comparison with our September diffusion run (volta-diffusion: Dream-7B Q8_0 in llama.cpp, 274 ms per denoising
  step, no KV cache, full 512-token sequence): d3LLM's ~105 ms per forward (2318 forwards in 243.7 s at 256 tokens)
  is a different model, precision and cache setup, with Python overhead. Treat it as context, not a like-for-like
  speedup.

## Files
- `setup.sh`: venv copied from an existing torch 2.10 cu128 venv, plus transformers 4.49.0, datasets 3.2.0,
  accelerate 1.10.1, the bundled lm-eval 0.4.9.1 with `[api]`, and einops 0.8.1.
- `patch_eval.py`, `eval_llada_v100.patch`
- `run_d3llm.sh` (`GEN` sets gen_length and steps), `run_qwen.sh`, `run_qwen2.sh` (`MAXTOK`), `run_set.sh`
- `rescore.py` and `hand_checked.json`. Run `python3 rescore.py results <arm> [<arm>]`.
- `results/`: logs, lm_eval JSON, per-question samples, and `run_set.log`.

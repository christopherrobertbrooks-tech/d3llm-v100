# Review artefacts for the d3LLM experiment

These are the small scripts and outputs I used to check the numbers in this repo's README. They back the findings filed as issues #1 to #6 at https://github.com/christopherrobertbrooks-tech/d3llm-v100/issues. Nothing here changes the experiment itself. Everything lives in the `review/` folder of the `review/main` branch.

Base: commit `a4f8315` on `main` (its tree hash is `4116d51`). This branch only adds `review/`.

## How to rerun

No GPU is needed. Every script reads the logs and samples under `results/` and prints text. Run them from anywhere:

    python3 review/scripts/recompute.py      # needs only the Python standard library
    python3 review/scripts/look_wrong.py     # standard library only
    python3 review/scripts/rescore.py        # needs transformers (tokenizer only)
    python3 review/scripts/tok.py            # needs transformers (tokenizer only)

The scripts find `results/` relative to their own location. To point them somewhere else, set `D3_RESULTS` to a results directory. `rescore.py` and `tok.py` load the d3LLM tokenizer, which is not in this repo. They default to `/mnt/nvme/models/d3llm/d3LLM_LLaDA` (the path on ember-gateway), and you can override it with `D3LLM_MODEL`. I ran them with the d3llm-sm70 venv (transformers 4.49.0), with `CUDA_VISIBLE_DEVICES` empty.

I changed one thing from my original copies: the hardcoded `/mnt/nvme/d3llm-sm70/results/` and tokenizer paths became the relative path and environment variables above. The logic is untouched. The files in `review/outputs/` were regenerated from this checkout, and the rescore output matches my earlier run line for line.

## Which file backs which issue

- **Issue #1 (scorer misreads answers):** `scripts/rescore.py` writes `outputs/rescore.out`. It re-scores each response using the boxed answer or the final sentence, and it also reports a union with lm_eval's flexible-extract. `scripts/look_wrong.py` writes `outputs/look_wrong.out` with examples where the right number is in the answer's last 150 characters but the scorer picked another number. `scripts/tok.py` writes `outputs/tok.out` with counts of wrong answers that contain the target number near the end (16 of 25 for Qwen at 512 tokens, 8 of 26 for d3LLM). `scripts/recompute.py` confirms lm_eval's flexible-extract scores of 74 (d3LLM fp16) and 75 (Qwen 512).
- **Issue #2 (VRAM):** `outputs/recompute.out` shows the "GPU Memory Used" lines in the d3LLM logs, which print an average over two GPUs. The comparison I quote in the issue (about 19 GiB for d3LLM against 5.1 GiB for Qwen) comes from the sentry log on the gateway. That log is not in my review folder, so nothing here backs that figure yet.
- **Issue #3 (token budgets):** `outputs/recompute.out` gives the Qwen runs at 256 and 512 tokens, how many responses hit the cap, and the accuracy of the capped ones. `outputs/rescore.out` and `outputs/tok.out` count d3LLM responses of 250 tokens or more (37 of 100, with a cap of 256). No d3LLM run at 512 exists, so there is no output for that.
- **Issue #4 (chat template):** I checked this by reading the prompts and run scripts. There is no script or output for it.
- **Issue #5 (llama-bench numbers):** I checked this by reading the committed `*-bench.log` files against the README. There is no script or output for it.
- **Issue #6 (smaller wording fixes):** `outputs/recompute.out` gives ms per forward pass (105.1 for d3LLM fp16), tokens per forward pass (8.98), seconds per question for both models, and the tokens-per-second figures. `outputs/tok.out` gives token counts in the LLaDA tokenizer next to Qwen's own counts, which is why tok/s is not like for like.

Also in `outputs/recompute.out`: the fp16 against bf16 comparison (identical extracted answer in 72 of 100, both right 67, fp16 only 7, bf16 only 5) and the KV cache against no-KV-cache comparison. The stop-rule checks (GPU 1 only, clock lock, free RAM, swap) were done by reading logs by hand, and no script from this folder covers them.

## Read this about the 87 and 77 numbers

The 87 (Qwen) and 77 (d3LLM) in issue #1 come from `rescore.py`. That is a rough heuristic I wrote quickly: it takes the boxed answer, or the numbers in the last line of the response. It is not a verified score. I did not check the individual responses by hand, and it is probably wrong for some. Use it as evidence that the original scorer's 75 against 74 is misleading, not as a replacement result. A proper rescore needs a real answer extractor applied to both models and a spot check by hand.

## Other notes

- `recompute.py` skips the first 10 requests as warmup when it computes the ms per forward pass, and it also prints a version that includes them.
- The per-request time for d3LLM is derived from cumulative times in the log, so it is only as exact as that log.
- I did not modify anything in `/mnt/nvme/d3llm-sm70` and did not use the GPU.

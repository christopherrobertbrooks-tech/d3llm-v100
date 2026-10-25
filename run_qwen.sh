#!/bin/bash
# AR baseline: Qwen2.5-7B-Instruct Q4_K_M in llama.cpp (sm_70 build, 5e03bdd), same GSM8K task through lm_eval with
# its chat template, greedy, 256 tokens. Plus llama-bench. usage: run_qwen.sh LIMIT TAG. Run under tools/v100.
set -u
LIMIT=$1; TAG=$2
W=/mnt/nvme/d3llm-sm70; B=/home/chris/llama.cpp-mainline/build-sm70/bin
M=/mnt/nvme/models/d3llm/qwen2.5-7b-instruct-gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
mkdir -p $W/results
echo "== qwen bench $(date +%r)"
CUDA_VISIBLE_DEVICES=1 $B/llama-bench -m $M -ngl 99 -p 512 -n 128 -r 3 > $W/results/$TAG-bench.log 2>&1; echo "bench rc=$?"
grep -E "pp512|tg128" $W/results/$TAG-bench.log
echo "== qwen gsm8k limit=$LIMIT $(date +%r)"
CUDA_VISIBLE_DEVICES=1 $B/llama-server -m $M -ngl 99 -c 4096 -np 1 --host 127.0.0.1 --port 8099 > $W/results/$TAG-server.log 2>&1 &
SP=$!
for i in $(seq 120); do curl -s 127.0.0.1:8099/health | grep -q ok && break; sleep 1; done
HF_ALLOW_CODE_EVAL=1 nice -n 10 $W/.venv/bin/python -m lm_eval --model local-chat-completions \
  --model_args model=qwen,base_url=http://127.0.0.1:8099/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
  --tasks gsm8k_cot_zeroshot --num_fewshot 0 --apply_chat_template --limit $LIMIT \
  --gen_kwargs temperature=0,max_gen_toks=256 --log_samples --output_path $W/results/$TAG > $W/results/$TAG.log 2>&1
echo "lm_eval rc=$? $(date +%r)"
kill $SP; wait $SP 2>/dev/null
grep -a -E "^\|.*(strict|flexible)" $W/results/$TAG.log | tail -2

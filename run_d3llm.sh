#!/bin/bash
# d3LLM-LLaDA on the V100: fp16, PyTorch SDPA (flash_attn not installed), multi-block + KV cache at the paper's
# GSM8K settings (eval_scripts/llada_gsm8k_cot.sh). usage: run_d3llm.sh LIMIT TAG [generation_method]
# Run under tools/v100. Speed = d3LLM's own show_speed (useful tokens / wall time, first 10 requests skipped).
set -u
LIMIT=$1; TAG=$2; METHOD=${3:-generate_multi_block_kv_cache}
W=/mnt/nvme/d3llm-sm70; mkdir -p $W/results
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "== d3llm $TAG limit=$LIMIT method=$METHOD $(date +%r) RAM available ${avail} GiB"
[ "$avail" -ge 9 ] || { echo "STOP: under 9 GiB RAM available"; exit 3; }
cd $W/d3LLM/utils/utils_LLaDA
EXTRA=""
[ "$METHOD" = generate_multi_block_kv_cache ] && EXTRA=",cache_delay_iter=2,refresh_interval=3"
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true CUDA_VISIBLE_DEVICES=1 D3_DTYPE=${D3_DTYPE:-float16} \
nice -n 10 $W/.venv/bin/python eval_llada_v100.py --tasks gsm8k_cot_zeroshot --num_fewshot 0 \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args model_path=/mnt/nvme/models/d3llm/d3LLM_LLaDA,gen_length=${GEN:-256},steps=${GEN:-256},block_length=32,show_speed=True,threshold=0.5,task=gsm8k_cot_zeroshot,generation_method=$METHOD,early_stop=True$EXTRA \
  --batch_size 1 --limit $LIMIT --log_samples --output_path $W/results/$TAG > $W/results/$TAG.log 2>&1
echo "rc=$? $(date +%r)"
grep -a -E "Throughput|Tokens per forward|Total time taken" $W/results/$TAG.log | tail -3
grep -a -E "^\|.*(strict|flexible)" $W/results/$TAG.log | tail -2

#!/bin/bash
cd /mnt/nvme/d3llm-sm70
./run_d3llm.sh 100 d3llm-fp16-kvcache
./run_qwen.sh 100 qwen-q4km
D3_DTYPE=bfloat16 ./run_d3llm.sh 100 d3llm-bf16-kvcache
./run_d3llm.sh 100 d3llm-fp16-multiblock generate_multi_block
echo "== set done $(date +%r)"

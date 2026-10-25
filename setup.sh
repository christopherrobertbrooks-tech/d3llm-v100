#!/bin/bash
set -euo pipefail
cd /mnt/nvme/d3llm-sm70
echo "== copy venv $(date +%r)"
[ -x .venv/bin/python ] || cp -a /mnt/nvme/humming/.venv .venv
rm -f .venv/lib/python3.12/site-packages/__editable__*humming* .venv/lib/python3.12/site-packages/_humming*.pth 2>/dev/null || true
printf "torch==2.10.0+cu128\ntriton==3.6.0\n" > constraints.txt
echo "== install $(date +%r)"
nice -n 10 ~/.local/bin/uv pip install -q --python .venv/bin/python -c constraints.txt \
  transformers==4.49.0 datasets==3.2.0 accelerate==1.10.1 torchdata==0.11.0 -e ./d3LLM/utils/lm-evaluation-harness
.venv/bin/python -c "import torch,transformers,accelerate,datasets,lm_eval;print(\"torch\",torch.__version__,\"tf\",transformers.__version__,\"acc\",accelerate.__version__,\"ds\",datasets.__version__,\"lm_eval\",lm_eval.__version__)"
python3 - <<PY
import importlib.util; print("flash_attn installed:", importlib.util.find_spec("flash_attn") is not None) 
PY
echo "== done $(date +%r)"

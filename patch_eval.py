s = open("eval_llada.py").read()
assert s.count("torch_dtype=torch.bfloat16,") == 2
s = s.replace("torch_dtype=torch.bfloat16,",
              'torch_dtype=getattr(torch, os.environ.get("D3_DTYPE", "float16")),  # V100: no bf16 tensor cores')
old = """        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})"""
assert old in s
s = s.replace(old, old + """
        else:  # V100 run: load straight onto the GPU (the gateway has 16 GB of RAM)
            model_kwargs.update({'device_map': {'': 'cuda:0'}, 'low_cpu_mem_usage': True})""")
open("eval_llada_v100.py", "w").write(s)
print("written")

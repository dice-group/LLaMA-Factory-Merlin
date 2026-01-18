1) Create and activate the conda env:
```bash
conda env create -f environment.yml
conda activate merlin
```

2) Install requirements separately:
```bash
pip install -r requirements.txt \
  --index-url https://pypi.org/simple \
  --no-cache-dir
```

3) insatlll flash attention seperately
```
pip install flash-attn --no-build-isolation --no-cache-dir
```

4) Install the project in editable mode without deps:
```bash
pip install -e . --no-deps
```

5) Do not install PEFT separately. It is vendored under `src/peft` and is
picked up by the editable install. `pip install -e src/peft` will fail because
there is no `setup.py` there.

Verify:
```bash
python -c "import torch, transformers, peft"
python -c "import llamafactory"
python -m llamafactory.launcher --help
```


## Eval environment
I decided to create a separate lm eval conda environemnt to prevent confliclts, since i didnt wanted to chagne transformer version in main conda environment after everything was implemented to prevent unexpected behaviour.
Thereofre use environment_eval.yml and install it. You may have to manually install the pip depdendecies like this for the cluster 

on cluster i had issues and finally installed like this:
```
conda create -y -n lm_eval \
  -c pytorch -c nvidia -c conda-forge \
  python=3.10 \
  pytorch=2.5.1 \
  pytorch-cuda=12.1 \
  cuda-toolkit=12.1 \
  omegaconf \
  pip
```
and then

```
pip install \
  --index-url https://pypi.org/simple \
  --no-cache-dir \
  transformers>=4.57.3 \
  lm-eval \
  "datasets<4" \
  accelerate \
  safetensors \
  wandb
```

check with 
```
python <<'PY'
import torch
import transformers
import lm_eval

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("lm_eval", lm_eval.__version__)
PY
```
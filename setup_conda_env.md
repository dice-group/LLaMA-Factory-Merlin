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

3) Install the project in editable mode without deps:
```bash
pip install -e . --no-deps
```

4) Do not install PEFT separately. It is vendored under `src/peft` and is
picked up by the editable install. `pip install -e src/peft` will fail because
there is no `setup.py` there.

Verify:
```bash
python -c "import torch, transformers, peft"
python -c "import llamafactory"
python -m llamafactory.launcher --help
```

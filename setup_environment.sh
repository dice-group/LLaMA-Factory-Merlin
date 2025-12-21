conda env create -f environment.yml || conda env update -f environment.yml
conda activate merlin
pip install flash-attn --no-build-isolation

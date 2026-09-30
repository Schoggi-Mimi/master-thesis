python -m pip install --upgrade pip setuptools wheel

python -m pip install `
  torch==2.1.0 `
  torchvision==0.16.0 `
  torchaudio==2.1.0 `
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install `
  mmcv==2.1.0 `
  --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

python -m pip install -r requirements-windows-full.txt
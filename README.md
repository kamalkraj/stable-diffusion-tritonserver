# stable-diffusion-tritonserver

Please checkout branch [v2](https://github.com/kamalkraj/stable-diffusion-tritonserver/tree/v2) for converting new models


## Download models
```bash
# clone this repo
git clone https://github.com/kamalkraj/stable-diffusion-tritonserver.git
cd stable-diffusion-tritonserver
# clone model repo from huggingface
git lfs install
git clone https://huggingface.co/kamalkraj/stable-diffusion-v1-4-onnx
```

Unzip the model weights
```bash
cd stable-diffusion-v1-4-onnx
tar -xvzf models.tar.gz
```


## Triton Inference Server

### Build
```bash
docker build -t tritonserver .
```

### Run
```
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 16384m   \
-v $PWD/stable-diffusion-v1-4-onnx/models:/models tritonserver \
tritonserver --model-repository /models/
```


## Inference

Install `tritonclient` and run the [notebook](Inference.ipynb) for inference.
```bash
pip install "tritonclient[http]"
```

## Credits
- ONNX conversion script from - [harishanand95/diffusers](https://github.com/harishanand95/diffusers/blob/dml/examples/inference/save_onnx.py)

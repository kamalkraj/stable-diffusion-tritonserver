# stable-diffusion-tritonserver


## Download models
```bash
git clone https://github.com/kamalkraj/stable-diffusion-tritonserver.git
cd stable-diffusion-tritonserver
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

Install `tritonclient` and run the [notebook](Inference.ipynb)
```bash
pip install "tritonclient[http]"
```

# stable-diffusion-tritonserver


## Download models
```bash
# clone this repo
git clone https://github.com/kamalkraj/stable-diffusion-tritonserver.git
cd stable-diffusion-tritonserver
```

## Build
```bash
# Build Docker
docker build -t  sd_trt .
```

## Convert to TensorRT
```bash
# run the conversion
docker run --gpus device=0 \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           -v $(pwd)/engine:/workspace/TensorRT/demo/Diffusion/engine \
           -v $(pwd)/onnx:/workspace/TensorRT/demo/Diffusion/onnx \
           -v $(pwd)/output:/workspace/TensorRT/demo/Diffusion/output \
           -it --rm sd_trt
export HF_TOKEN=<your access token>
cd demo/Diffusion/
LD_PRELOAD=${PLUGIN_LIBS} python3 demo-diffusion.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN -v

# Exit from the Docker and Move the model weights to tritonserver folder 
bash copy_files.sh
```

## Triton Inference Server

### Run
```bash
docker run -it --rm --gpus device=0 -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 16384m   \
-v $PWD/models:/models sd_trt bash
# run the server
LD_PRELOAD=${PLUGIN_LIBS} CUDA_MODULE_LOADING=LAZY tritonserver --model-repository /models/ --model-control-mode=explicit
```


## Inference

Install `tritonclient` and run the [notebook](Inference.ipynb) for inference.
```bash
pip install "tritonclient[http]==2.28.0"
```

## Credits
- TRT conversion script from - [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT/tree/main/demo/Diffusion).
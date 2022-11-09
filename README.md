# stable-diffusion-tritonserver


## Download models
```bash
# clone this repo
git clone https://github.com/kamalkraj/stable-diffusion-tritonserver.git
cd stable-diffusion-tritonserver
```

## Install
```bash
# create a virtualenv
virtualenv env
# activate
source env/bin/activate
# upgrade pip
pip install -U pip
# install libs
pip install -r requirements.txt
```

## Convert to onnx
```bash
# run the conversion
python convert_stable_diffusion_checkpoint_to_onnx.py --model_path nitrosocke/mo-di-diffusion--output_path stable-diffusion-onnx --opset 16 --fp16
# Move the model weights
bash copy_files.sh
```


## Triton Inference Server

### Build
```bash
docker build -t tritonserver .
```

### Run
```
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 16384m   \
-v $PWD/models:/models tritonserver \
tritonserver --model-repository /models/
```


## Inference

Install `tritonclient` and run the [notebook](Inference.ipynb) for inference.
```bash
pip install "tritonclient[http]"
```

## Credits
- ONNX conversion script from - [harishanand95/diffusers](https://github.com/harishanand95/diffusers/blob/dml/examples/inference/save_onnx.py) and [huggingface](https://github.com/huggingface/diffusers)

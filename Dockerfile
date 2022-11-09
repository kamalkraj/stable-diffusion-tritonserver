FROM nvcr.io/nvidia/tritonserver:22.10-py3

RUN pip install -U pip

RUN pip install torch==1.12.1

RUN pip install --upgrade git+https://github.com/huggingface/diffusers@0248541deadfa187150fe7f96a575ff905ecddd7 scipy==1.9.3 transformers==4.24.0
FROM nvcr.io/nvidia/tritonserver:22.12-py3

RUN pip install -U pip

RUN pip install torch==1.13.1

RUN pip install --upgrade diffusers==0.11.1 transformers==4.25.1
FROM nvcr.io/nvidia/tritonserver:22.08-py3

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip install --upgrade diffusers scipy transformers

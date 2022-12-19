FROM nvcr.io/nvidia/tritonserver:22.11-py3

WORKDIR /workspace

RUN apt-get update && apt-get install cmake -y

RUN pip install --upgrade pip && pip install --upgrade tensorrt

RUN git clone https://github.com/NVIDIA/TensorRT.git -b main --single-branch \
    && cd TensorRT \
    && git submodule update --init --recursive

ENV TRT_OSSPATH=/workspace/TensorRT
WORKDIR ${TRT_OSSPATH}

RUN mkdir -p build \
    && cd build \
    && cmake .. -DTRT_OUT_DIR=$PWD/out \
    && cd plugin \
    && make -j$(nproc)

ENV PLUGIN_LIBS="${TRT_OSSPATH}/build/out/libnvinfer_plugin.so"

RUN cd demo/Diffusion/ \
    && mkdir -p onnx engine output

RUN pip3 install -r demo/Diffusion/requirements.txt

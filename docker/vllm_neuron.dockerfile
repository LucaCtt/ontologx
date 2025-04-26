# Configurable environment variables
ENV MODEL="meta-llama/Llama-3.2-1B"
ENV TEMPERATURE="0.7"
ENV MAX_MODEL_LEN="4096"

FROM "public.ecr.aws/neuron/pytorch-inference-neuronx:2.5.1-neuronx-py310-sdk2.22.0-ubuntu22.04"

# Install some basic utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        python3 \
        python3-pip \
        ffmpeg libsm6 libxext6 libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /vllm

# Install neuronx sdk
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir fastapi ninja tokenizers pandas
RUN python3 -m pip install sentencepiece transformers==4.51.1 -U
RUN python3 -m pip install transformers-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com -U
RUN python3 -m pip inskall --pre neuronx-cc==2.* --extra-index-url=https://pip.repos.neuron.amazonaws.com -U

# Install vllm
ENV VLLM_TARGET_DEVICE="neuron"
RUN git clone https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    git checkout v0.8.4 && \
    python3 -m pip install -U \
        cmake>=3.26 ninja packaging setuptools-scm>=8 wheel jinja2 \
        -r requirements-neuron.txt && \
    pip install --no-build-isolation -v -e . && \
    pip install --upgrade triton==3.0.0

CMD ["vllm", "serve", "${PARSER_MODEL}", "--tensor-parallel-size", "2", "--max-model-len", "${MAX_MODEL_LEN}", "--temperature", "${TEMPERATURE}"]
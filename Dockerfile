FROM docker.smirrors.lcpu.dev/library/ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# 设置默认 shell 为 bash
SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y wget \
    && apt install -y vim

WORKDIR /cpm-bee

COPY . .

# 拷贝CANN
RUN mkdir -p /usr/local/Ascend/ \
    && mv /cpm-bee/ascend-toolkit/ /usr/local/Ascend/ascend-toolkit/ \
    && echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc

# 安装conda
RUN source ~/.bashrc \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
    && bash Miniconda3-latest-Linux-aarch64.sh -b -u -p /root/software/miniconda3 \
    && /root/software/miniconda3/bin/conda init

# 设置环境变量
ENV PATH=/root/software/miniconda3/bin:$PATH

# 创建 conda 环境
RUN source ~/.bashrc \
    && conda create --name cpm-bee python=3.9 \
    && echo "source activate cpm-bee" >> ~/.bashrc

# 安装 python 包
RUN source ~/.bashrc \
    && source activate cpm-bee \
    && pip install -r requirements.txt \
    && mkdir -p ~/.cache/huggingface/accelerate/ \
    && mv /cpm-bee/default_config.yaml ~/.cache/huggingface/accelerate/
# 介绍

对 cpm-bee 的数据结构稍加改造之后进行训练，并打包环境为 docker，在 docker 中测试 npu 上的训练和推理速度。如果不需要 docker 环境，直接参照 ![[说明文档]](cpm-bee.md) 进行本地安装。如果不需要 npu，在脚本中注释掉与 npu 相关的部分或改为对应的 cuda 命令。


# 拉取镜像

```bash
# 拉取基础镜像
docker pull docker pull ubuntu:20.04
# 或使用代理
docker pull docker.smirrors.lcpu.dev/library/ubuntu:20.04
```

# 构建镜像

```bash
# 复制 toolkit 包和 accelerate 配置
cp /usr/local/Ascend/ascend-toolkit . -rf
cp ~/.cache/huggingface/accelerate/default_config.yaml ./
# 删除上次构建的镜像
docker rmi cpm-bee
# 构建镜像
docker build -f Dockerfile -t cpm-bee:latest .
```

# 运行容器

```bash
# 模型路径 (cpm-bee-2b 和 cpm-bee-10 文件存放在此)
model_path="/home/liying/cmp-bee-finetune/models"

# 创建容器
sudo docker run -it -u root -d --ipc=host --network=host \
--device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
-v ${model_path}:/cpm-bee/models \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
--name cpm-bee-container cpm-bee
```

# 进入容器
sudo docker attach cpm-bee-container

在容器中执行
```bash
./test.sh
```

输出结果在 report.txt 中

# 安装 conda

```bash
# 确认系统架构
uname -m
# 安装
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh -b -u -p ~/software/miniconda3
# 初始化
~/software/miniconda3/bin/conda init
source ~/.bashrc
# 设定卡
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
```

## 安装环境
```bash
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 安装环境
conda create --name cpm-bee python=3.9
conda activate cpm-bee
pip install -r requirements.txt
```

# 推理

61机器，liying用户下测试

## 下载模型

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download openbmb/cpm-bee-2b --local-dir models/cpm-bee-2b
```

## 运行模型

单卡
```bash
python cpm-bee-infa.py --model-path models/cpm-bee-2b 2>&1
```

多卡
```bash
# 模型并行
python cpm-bee-infa-multi.py --model-path models/cpm-bee-10b --devices 4,5 > running.log 2>&1
```

使用 npu-smi info 查看 npu 运行情况

# 微调

修改 data_prepare.py 文件以读取数据文件

数据需处理为 trainset.__getitem__() 返回值为 {"input": "...", "<ans>": ""} 的形式

参考 https://github.com/OpenBMB/CPM-Bee/tree/main/tutorials/basic_task_finetune

单卡：
```bash
python cpm-bee-train.py --model-path models/cpm-bee-2b > running.log 2>&1
```

多卡：
```bash
# 设置分布式训练参数
accelerate config
# 训练
accelerate launch cpm-bee-train-accelerate.py > running.log 2>&1
```

使用 accelerate 的配置文件为：
( 文件在 ~/.cache/huggingface/accelerate/default_config.yaml)

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_NPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: 0,1,2,3
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

# 进行测试

```bash
/.test.sh
```

输出结果在 report.txt 中


# 测试结果

## 推理

2b单卡推理：8.89 tokens/second
10b单卡推理：4.86 tokens/second

10b 2卡:2.65 tokens/second

模型并行对10b是没有必要的，反而减慢推理速度


## 微调

2b单卡2batch每条1024token：0.2173 s/step
2b四卡2batch每条1024token：0.3607 s /step (四卡并行，每个 step 处理 2batch*4npu = 8 条数据)
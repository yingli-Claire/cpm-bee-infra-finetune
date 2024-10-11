# 推理

82机器，liying用户下测试

## 下载模型

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download openbmb/cpm-bee-2b --local-dir /share/liying/models/cpm-bee-2b
```

## 安装环境
```bash
conda create --name cpm-bee python=3.9
conda activate cpm-bee
cd cmp-bee
pip install -r requirements.txt --trusted-host mirrors.aliyun.com
```

## 运行模型

```bash
python cpm-bee-infa.py
```

# 微调

修改 data_prepare.py 文件以读取数据文件

数据需处理为 trainset.__getitem__() 返回值为 {"input": "...", "<ans>": ""} 的形式

参考 https://github.com/OpenBMB/CPM-Bee/tree/main/tutorials/basic_task_finetune

```bash
# 设置分布式训练参数
accelerate config
# 训练
accelerate launch cpm-bee-train.py > running_log 2>&1
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

# DeepSpeed

还可以使用 deepspeed 优化 (npu 支持没有跑通):

安装 

```bash
git clone https://gitee.com/ascend/DeepSpeed.git
cd DeepSpeed
pip install ./
```


```bash
# 设置分布式训练参数
accelerate config
# 使用 deepspeed 优化后
accelerate launch cpm-bee-train_deepspeed.py > running2_log 2>&1
```

~/.cache/huggingface/accelerate/default_config.yaml 内容

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
 gradient_clipping: 1.0
 zero_stage: 2
distributed_type: DEEPSPEED
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
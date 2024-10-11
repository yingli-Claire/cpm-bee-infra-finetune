from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator, DeepSpeedPlugin
from torch.utils.data import DataLoader
from data_prepare import CPMDataset

import torch
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)

import time

torch.npu.empty_cache()

# 数据准备
trainset = CPMDataset("/home/liying/cpm-bee-infer_and_finetune/basic_task_finetune/bee_data/train.jsonl")  
train_loader = DataLoader(trainset, batch_size=1)

# 模型
model_path = "/share/liying/models/cpm-bee-2b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

optimizer = torch.optim.Adam(model.parameters())

# DeepSpeed 配置文件路径
deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
if accelerator.is_main_process:
    print(f"Using DeepSpeed config: {deepspeed_plugin.deepspeed_config}")
    
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

# 计时
total_time = 0

for iter, data in enumerate(train_loader):
    
    model.train()
    
    step_start = time.perf_counter()
    
    # 训练
    optimizer.zero_grad()
    input_encoded = tokenizer.prepare_for_finetune(data, max_length=512)
    outputs = model(**input_encoded)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    
    step_time = time.perf_counter() - step_start
    total_time += step_time
    
    # 仅在主进程输出
    if accelerator.is_main_process:
        print(f"Step {iter}, Loss: {loss.item():.4f}, Time per step: {step_time:.4f} s")

    
# 仅在主进程输出
if accelerator.is_main_process:
    print("Training done")
    print(f"Total training time: {total_time:.2f} s")
    print(f"Average time per step: {total_time / (iter + 1):.4f} s")
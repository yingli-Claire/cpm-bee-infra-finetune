# 设置命令行参数解析
import argparse
parser = argparse.ArgumentParser(description='Run model with specified devices and model path.')
parser.add_argument('--model-path', type=str, required=True, help='Path to the model directory')
args = parser.parse_args()

from transformers import AutoTokenizer, AutoModelForCausalLM
from data_prepare import CPMDataset, convert_to_list
from accelerate import Accelerator
import torch
import torch_npu
from torch.utils.data import DataLoader
import time
torch.npu.empty_cache()

# 数据准备
trainset = CPMDataset("basic_task_finetune/bee_data/train.jsonl")
#trainset = trainset[:100]
train_loader = DataLoader(trainset, batch_size=4, shuffle=True)

# 模型
model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# 准备模型和优化器
optimizer = torch.optim.Adam(model.parameters(), eps=1e-4)
accelerator = Accelerator()
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

# 计时
total_time = 0

model.train()
for epoch in range(2):
    for iter, data in enumerate(train_loader):
        
        data = convert_to_list(data)
        
        step_start = time.perf_counter()
        
        # 训练    
        optimizer.zero_grad()
        input_encoded = tokenizer.prepare_for_finetune(data, max_length=1024).to(model.device)
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
    print(f"Average time per step: {total_time / (iter + 1)*epoch:.4f} s")
    print(f"TESTED: {total_time / (iter + 1)*epoch:.4f} s / step")

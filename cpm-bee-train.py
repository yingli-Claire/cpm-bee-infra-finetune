# 设置命令行参数解析
import argparse
parser = argparse.ArgumentParser(description='Run model with specified devices and model path.')
parser.add_argument('--model-path', type=str, required=True, help='Path to the model directory')
args = parser.parse_args()

from transformers import AutoTokenizer, AutoModelForCausalLM
from data_prepare import CPMDataset
import torch
import torch_npu
import time
torch.npu.empty_cache()

# 数据准备
trainset = CPMDataset("basic_task_finetune/bee_data/train.jsonl")
trainset = trainset[:100]  

# 模型
model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to('npu')

# 训练
optimizer = torch.optim.Adam(model.parameters(), eps=1e-4)

# 计时
total_time = 0

batch = 2
model.train()
for i in range(0, len(trainset), batch):
    data = trainset[i:i+batch]
    
    step_start = time.perf_counter()
    
    # 训练
    optimizer.zero_grad()
    input_encoded = tokenizer.prepare_for_finetune(data, max_length=1024).to(model.device)
    outputs = model(**input_encoded)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    step_time = time.perf_counter() - step_start
    total_time += step_time
    
    print(f"Step {i//batch}, Loss: {loss.item():.4f}, Time per step: {step_time:.4f} s")

print("Training done")
print(f"Total training time: {total_time:.2f} s")
print(f"Average time per step: {total_time / ((i // batch) + 1):.4f} s")

print(f"TESTED: {total_time / ((i // batch) + 1):.4f} s / step")
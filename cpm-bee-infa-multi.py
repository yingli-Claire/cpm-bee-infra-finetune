# 设置命令行参数解析
import argparse
parser = argparse.ArgumentParser(description='Run model with specified devices and model path.')
parser.add_argument('--devices', type=str, required=True, help='Set visible devices for ASCEND_RT_VISIBLE_DEVICES')
parser.add_argument('--model-path', type=str, required=True, help='Path to the model directory')
args = parser.parse_args()

# 设置环境变量
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.devices

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model
from accelerate.utils import get_balanced_memory, infer_auto_device_map
import time
import torch
import torch_npu


# 模型路径
model_path = args.model_path

# 分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 获取每个设备的最大内存容量
max_memory = get_balanced_memory(
    model, 
    no_split_module_classes=["CpmBeeTransformerBlock"]
)
# 打印每个设备的最大内存容量
print("Max memory allocation for each device:")
for device, memory in max_memory.items():
    print(f"Device {device}: {memory / (1024**2):.2f} MB")

# 设备映射字典
device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["CpmBeeTransformerBlock"]) 
# 确保二者在同一个设备上
# make sure the data on the same device when projecting hidden states to logits.
device_map["cpmbee.encoder.output_layernorm"] = device_map["cpmbee.input_embedding"] = 0
# 打印设备映射
for module, device in device_map.items():
    print(f"Module {module} is allocated to device {device}")

# 分配模型到设备
model = dispatch_model(model, device_map=device_map)

# 生成
start_time = time.time()
res = model.generate(
    [{"input": "NGC 6231是一个位于天蝎座的疏散星团，天球座标为赤经16时54分，赤纬-41度48分，视觉观测大小约45角分，亮度约2.6视星等，距地球5900光年。NGC 6231年龄约为三百二十万年，是一个非常年轻的星团，星团内的最亮星是5等的天蝎座 ζ1星。用双筒望远镜或小型望远镜就能看到个别的行星。NGC 6231在1654年被意大利天文学家乔瓦尼·巴蒂斯特·霍迪尔纳（Giovanni Battista Hodierna）以Luminosae的名字首次纪录在星表中，但是未见记载于夏尔·梅西耶的天体列表和威廉·赫歇尔的深空天体目录。这个天体在1678年被爱德蒙·哈雷（I.7）、1745年被夏西亚科斯（Jean-Phillippe Loys de Cheseaux）（9）、1751年被尼可拉·路易·拉卡伊（II.13）分别再次独立发现。", "prompt": "中翻英", "<ans>": ""}],
    tokenizer,
    max_new_tokens=1000,
    min_length=500,
    # num_beams=1,         # 使用beam search来提高生成质量
    # early_stopping=True  # 启用提前停止
)
end_time = time.time()

# calculate speed
num_generated_tokens = 0
for output in res:
    text = output['<ans>']
    tokens = tokenizer(text, return_tensors='pt')["input_ids"]
    num_tokens = tokens.shape[1]
    num_generated_tokens += num_tokens

time_taken = end_time - start_time
tokens_per_second = num_generated_tokens / time_taken
print(f"Generated {num_generated_tokens} tokens in {time_taken:.2f} seconds ({tokens_per_second:.2f} tokens/second)")
print(f"TESTED: {tokens_per_second:.2f} tokens/second")

print(res[0]['<ans>'])
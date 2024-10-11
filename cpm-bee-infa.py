from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_npu
import time

torch_npu.npu.set_compile_mode(jit_compile=False)

# 模型路径
model_path = "/share/liying/models/cpm-bee-2b"

# 分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 模型
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16,
    trust_remote_code=True
).to('npu')

# 生成
start_time = time.time()
res = model.generate(
    {"input": "NGC 6231是一个位于天蝎座的疏散星团，天球座标为赤经16时54分，赤纬-41度48分，视觉观测大小约45角分，亮度约2.6视星等，距地球5900光年。NGC 6231年龄约为三百二十万年，是一个非常年轻的星团，星团内的最亮星是5等的天蝎座 ζ1星。用双筒望远镜或小型望远镜就能看到个别的行星。NGC 6231在1654年被意大利天文学家乔瓦尼·巴蒂斯特·霍迪尔纳（Giovanni Battista Hodierna）以Luminosae的名字首次纪录在星表中，但是未见记载于夏尔·梅西耶的天体列表和威廉·赫歇尔的深空天体目录。这个天体在1678年被爱德蒙·哈雷（I.7）、1745年被夏西亚科斯（Jean-Phillippe Loys de Cheseaux）（9）、1751年被尼可拉·路易·拉卡伊（II.13）分别再次独立发现。", "prompt": "中翻英", "<ans>": ""},
    tokenizer,
    max_new_tokens=1000,
    min_length=50,
)
end_time = time.time()

# calculate speed
num_generated_tokens = 0
for output in res:
    text = output['<ans>']
    tokens = tokenizer(text, return_tensors='pt')["input_ids"]
    num_tokens = tokens.shape[1]
    num_generated_tokens += num_tokens
    print(f"The text '{text}' has {num_tokens} tokens.")
time_taken = end_time - start_time
tokens_per_second = num_generated_tokens / time_taken
print(f"Generated {num_generated_tokens} tokens in {time_taken:.2f} seconds ({tokens_per_second:.2f} tokens/second)")

print(res)
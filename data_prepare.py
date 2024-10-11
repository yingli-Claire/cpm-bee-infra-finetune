import json
from torch.utils.data import Dataset

class CPMDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line in file:
                # 解析每一行 JSON 数据
                item = json.loads(line)
                # 提取需要的字段
                input_text = item['input']
                answer = item['<ans>']
                # 将数据添加到列表中
                self.data.append({"input": input_text, "<ans>": answer})

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, idx):
        # 返回格式化的数据
        return self.data[idx]

if __name__=="__main__":

    dataset = CPMDataset('eval.jsonl')

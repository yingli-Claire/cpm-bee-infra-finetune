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
                # input = item['input']
                # options = item['options']
                # question = item['question']
                # answer = item['<ans>']
                # input_text = f"{input}<sep>{question}<sep>{options}"
                # 将数据添加到列表中
                self.data.append(item)

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, idx):
        # 返回格式化的数据
        return self.data[idx]
    
def convert_to_list(data):
    
    data2 = []
    n = len(data['<ans>'])
    keys = data.keys()
    key2s = data['options'].keys()
    for i in range(n):
        sample = {}
        for key in keys:
            if key == 'options':
                sub_sample = {}
                for key2 in key2s:
                    sub_sample[key2] = data[key][key2][i]
                sample[key] = sub_sample
            else:
                sample[key] = data[key][i]
        data2.append(sample)
        
    return data2

if __name__=="__main__":

    from torch.utils.data import DataLoader
    
    trainset = CPMDataset("basic_task_finetune/bee_data/eval.jsonl")
    trainset = trainset[:100]
    train_loader = DataLoader(trainset, batch_size=2, shuffle=True)
    
    for iter, data in enumerate(train_loader):
        data = convert_to_list(data)
        print(data)
        break
#!/bin/bash

# 单卡推理
python cpm-bee-infa.py --model-path models/cpm-bee-2b > 2b-1npu-infer.log 2>&1
python cpm-bee-infa.py --model-path models/cpm-bee-10b > 10b-1npu-infer.log 2>&1

# 多卡推理
python cpm-bee-infa-multi.py --model-path models/cpm-bee-10b --devices 0,1 > 10b-2npu-infer.log 2>&1

# 单卡微调
python cpm-bee-train.py --model-path models/cpm-bee-2b > 2b-1npu-fintune.log 2>&1

# 多卡微调
accelerate launch cpm-bee-train-accelerate.py --model-path models/cpm-bee-2b > 2b-4npu-fintune.log 2>&1



# 报告生成
# 报告生成
{
  echo "2b单卡推理:"
  grep TESTED 2b-1npu-infer.log

  echo "10b单卡推理:"
  grep TESTED 10b-1npu-infer.log

  echo "10b两卡推理:"
  grep TESTED 10b-2npu-infer.log

  echo "2b单卡微调:"
  grep TESTED 2b-1npu-fintune.log

  echo "2b四卡微调:"
  grep TESTED 2b-4npu-fintune.log
} > report.txt




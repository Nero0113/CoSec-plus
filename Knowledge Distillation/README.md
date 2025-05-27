## 1 Environment
pip3 install transformers\==4.18.0
pip3 install torch\==2.0.1
pip3 install deepspeed\==0.10.0
pip3 install torchvision\==0.15.2
pip3 install nltk
pip3 install numerize
pip3 install rouge-score
pip3 install torchtyping
pip3 install rich
pip3 install accelerate
pip3 install datasets
pip3 install sentencepiece
pip3 install protobuf\==3.20.3
pip3 install peft

## 2 Data
### 2.1 Resources
- The training intruction-response data before processing can be downloaded from this [link]([ise-uiuc/Magicoder-Evol-Instruct-110K · Datasets at Hugging Face](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K))
### 2.2 Data Processing
- Tokenize the data and store them in binary files:
```bash
bash scripts/codegen/tools/process_data_magicoder_Evol.sh
```

## 3 Distill the Sec models
```bash
# codegen
bash scripts/codegen/kd/kd_350MB_6B_fkl.sh
# deepseek-coder
bash scripts/deepseek-coder/kd/kd_1.3B_6.7B_rkl.sh
# qwen-2.5 need two A800
bash scripts/Qwen-2.5/kd/kd_1.5B_7B_two_gpus.sh
# starcoder
bash scripts/starcoder/kd/kd_1B_7B_rkl.sh
```

**We provide three distilled security model bases.** You can download it from Google Drive: 

https://drive.google.com/drive/folders/1j0kf1lBDwK9AmUw1JVQ0RhiIDjHiDLY3?usp=drive_link

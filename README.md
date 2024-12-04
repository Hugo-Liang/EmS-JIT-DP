# Empirical Study of Just-In-Time Defect Prediction
Replication package for the paper entitled: An Extensive Study on Pre-Trained Models for Just-in-Time Defect Prediction

### Data Preparation
The `Original` and `Unified` datasets are can be found [here](https://drive.google.com/drive/folders/1vjaJGEYyHIVq7ZPq3P_7N_Pc6_q7slS7?usp=sharing). Download `*.tar.gz` and extract it under the `Dataset/fine-tuning/JITDefectPrediction/` folder via `tar -zxvf *.tar.gz`.


### Pretrained model files Preparation
Manually download the **added_tokens.json, config.json, merges.txt, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, and vocab.json** from [Hugging Face-CodeT5](https://huggingface.co/Salesforce/codet5-base/tree/main), upload them to **models/codet5_base/**.


### Environment Settings
* GPU: Nvidia Tesla P40
* OS: CentOS 7.6

```
git clone https://github.com/Hugo-Liang/SETCS.git
cd SETCS
conda create -n EmS python=3.8
conda activate EmS
pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Run
#### CodeT5 with random forest (JIT-Core)

**1.1 CodeT5 with semantic feature**

```bash scripts/finetune_jitdp_SF.sh -g 0```

**1.2. Random forest with semantic feature**

```python RF.py```

**1.3. Average prediction results from CodeT5 and Forest**

```python combine.py```


#### CodeT5 with concatenated embeddings (JIT-Coca)

**2. CodeT5 with semantic feature and expert features**

```bash scripts/finetune_jitdp_SF_EF.sh -g 0```


### Get Involved
Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports.


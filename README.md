<div align="center">

#  MCM-DPO
</div>

This is the Source Code of Paper: **MCM-DPO: Multifaceted Cross-Modal Direct Preference Optimization for Alter-text Generation**



## Brief Introduction 

1. **Main Motivation**

The task of alternative text (alt-text) generation aims to produce concise, context-dependent textual descriptions of images, which has achieved state-of-the-art performance with current multimodal large language models (MLLMs) using supervised fine-tuning (SFT).  
However, existing methods are limited by the challenges of SFT training and issues inherent in alt-text data, preventing further performance improvements.  


<p align="center" width="100%">
<a target="_blank"><img src="./images/performance_gap.png" alt="performance_gap" style="width: 50%; min-width: 200px; display: block; margin: auto;"></a>
</p>

 <!-- ![framework](./images/performance_gap.png) -->

2. **The Framework of  MCM-DPO**
To address these challenges, we propose a novel approach, Multifaceted Cross-Modal Direct Preference Optimization (MCM-DPO), which optimizes preferences across Single Preference, Pairwise Preference, and Multi-Preference dimensions, covering text, image, and cross-modal factors.  

<p align="center" width="100%">
<a target="_blank"><img src="./images/framework.png" alt="framework" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>


 <!-- ![framework](./images/framework.png) -->


3. **Exploration of Training Paradigms** MLLMs comprise a vision encoder, LLMs, and a projection. Current works freeze the vision encoder, training only the LLMs and projection. This limits cross-modal integration, leading to hallucinations and errors. To assess how training paradigms affect MLLMs in new domains (e.g., social media) and tasks, we explore four paradigms (See the following figure). Training occurs in two stages: (1) supervised fine-tuning (SFT) on large datasets and (2) preference optimization (DPO or MCM-DPO) on smaller high-quality datasets. The explored paradigms are:


<p align="center" width="100%">
<a target="_blank"><img src="./images/paradigm.png" alt="paradigm" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>
 <!-- ![framework](./images/paradigm.png) -->

For more technical details, kindly refer to the our paper.

## Contents
- [1. Preparing Dataset](#data)
- [2. Environment Preparation](#install)
- [3. Training](#training)
- [4. Evaluation](#evaluation)


## 1. Preparing Dataset

### Training data
To collect a high-quality dataset of human preferences for building an automatic alt-text generation system, we gathered human-annotated alt-text datasets from two widely used social media platforms: Twitter and Pinterest, our dataset construction process involves three steps: 
* (1) Collection of User-written Data; 
* (2) Grammar Correction; 
* (3) Preference Collection. 


Sample data from Twitter and Pinterest are provided in the `./data/twitter` and  `./data/pinterest`.

### Evaluation data

The evaluation data can be downloaded from [here](https://drive.google.com/) and placed in the playground/data directory


## 2. Environment Preparation

1. Clone this repository and navigate to source folder
```bash
cd MCMDPO
```

2. Build Environment 


```Shell


echo "Creating conda environment"
conda create -n mcmdpo python=3.10
conda activate mcmdpo

echo "Installing dependencies"
pip install -e .
```


## 3. Training
Firstly, configure the training dataset name data_path and the checkpoint name output_dir.

* #### LLaVA SFT/MCMDPO/DPO Training
```Shell

bash scripts/sft.sh
bash scripts/mcmdpo.sh
bash scripts/dpo.sh
```


## 4. Evaluation

1. Run inference to generate responses

```py
python mcmdpo_inference.py --model_name {ckpt_name} --test_datasets {test_datasets} --eval_output {eval_output} 
```


2. Evaluate the generated responses.

```py
python get_score.py --file {eval_output}
```


## Acknowledgement

Our MCM-DPO is developed based on the codebases of [LLaVA](https://github.com/haotian-liu/LLaVA), and we would like to thank the developers.
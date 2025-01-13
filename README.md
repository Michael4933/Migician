<p align="center">
<img src="figs/logo.png" style="width: 30%"> 
</p>

# Migician: Revealing the Magic of Free-Form Multi-Image Grounding in Multimodal Large Language Models
[You Li](https://scholar.google.com.hk/citations?user=RZ5bOS0AAAAJ&hl=zh-CN), [Heyu Huang](https://openreview.net/profile?id=~Heyu_Huang2)*, [Chen Chi](https://openreview.net/profile?id=~Chi_Chen1), [Kaiyu Huang](https://openreview.net/profile?id=~Kaiyu_Huang1), Chao Huang, Zonghao Guo, Zhiyuan Liu, Jinan Xu, Yuhua Li, Ruixuan Li, Maosong Sun

-----

<a href='https://michael4933.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='#'><img src='https://img.shields.io/badge/Demo-Page-purple'></a> 
<a href='https://arxiv.org/abs/2501.05767'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
<a href='https://huggingface.co/Michael4933/Migician'><img src='https://img.shields.io/badge/Model-Huggingface-red'></a> 
<a href='https://huggingface.co/datasets/Michael4933/MIG-Bench'><img src='https://img.shields.io/badge/Benchmark-Huggingface-yellow'></a> 
<a href='https://huggingface.co/datasets/Michael4933/MGrounding-630k'><img src='https://img.shields.io/badge/Dataset-Huggingface-blue'></a> 

This repository hosts the usage details of our training dataset <strong>MGrounding-630k</strong> and benchmark <strong>MIG-Bench</strong> and the training implementation of Migician, the first competitive Multi-image Grounding MLLM capable of free-form grounding.

-----------

## 📰 News
* **[2025.01.13]**  🌷🌷🌷 We have further released our massive multi-image grounding training dataset [MGrounding_630k](https://huggingface.co/datasets/Michael4933/MGrounding-630k) and our comprehensive multi-image grounding benchmark [MIG-Bench](https://huggingface.co/datasets/Michael4933/MIG-Bench) on Huggingface🤗~ Feel free to download and apply them for your own use.
* **[2025.01.12]**  🌟🌟🌟 The model weights are now available on HuggingFace! 🤗 Download and have a try at [Huggingface Model](https://huggingface.co/Michael4933/Migician)!
* **[2025.01.10]** 🌞🌞🌞 We have released our paper on [Arxiv](https://arxiv.org/abs/2501.05767) at the start of the new year!

## 📝 Abstract

The recent advancement of Multimodal Large Language Models (MLLMs) has significantly improved their fine-grained perception of single images and general comprehension across multiple images. However, existing MLLMs still face challenges in achieving precise grounding in complex multi-image scenarios. To address this, we first explore a Chain-of-Thought (CoT) framework that integrates single-image grounding with multi-image comprehension. While partially effective, it remains unstable and struggles to capture abstract visual information due to its non-end-to-end nature. Therefore, we introduce <strong>Migician</strong><img src="figs/logo.png" style="width: 3%">, the first multi-image grounding model capable of performing free-form and accurate grounding across multiple images. To support this, we present the [MGrounding-630k](https://huggingface.co/datasets/Michael4933/MGrounding-630k) dataset, which comprises data for several multi-image grounding tasks derived from existing datasets, along with newly generated free-form grounding instruction-following data. Furthermore, we propose [MIG-Bench](https://huggingface.co/datasets/Michael4933/MIG-Bench), a comprehensive benchmark specifically designed for evaluating multi-image grounding capabilities. Experimental results demonstrate that our model achieves significantly superior multi-image grounding capabilities, outperforming the best existing MLLMs by 21.61% and even surpassing much larger 70B models.
          

## 😮 Top Multi-Image Grounding Capacity
<p align="center">
<img src="figs/radar.png" width=100%>
</p>
<p align="center">
<img src="figs/multi_general.png" width=100%>
</p>
Migician surpasses much larger 70B scale model over all tasks on MIG-Bench by a great margin as shown in the radar image above. Additionally, it demonstrates great competitiveness in several general multi-image understanding benchmarks. We are looking forward to the promising applications of Migician on a broad spectrum of real-world scenarios.

## 👉 Getting Started
<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Environment'>1. Environment</a>
* <a href='#Data Preparation'>2. Data Preparation </a>
* <a href='#Inference and Evaluation'>3. Inference and Evaluation</a>
  * <a href='#Inference'>3.1. Inference</a>
  * <a href='#Evaluation'>3.2. Evaluation </a>
* <a href='#Finetune'>4. Finetune</a>
  

<span id='Environment'/>

### 1. Environment  <a href='#all_catelogue'>[Back to Top]</a>
Follow the commands below to establish a plausible environment.
```
conda env create -n migician python=3.10

git clone https://github.com/Michael4933/Migician.git
cd Migician

conda activate migician
pip install -r requirements.txt
```

<span id='Data Preparation'/>

### 2. Data Preparation <a href='#all_catelogue'>[Back to Top]</a>
MGrounding-630k encompasses a diverse collection of multi-image grounding tasks and numerous images from different sources. For convenient utilization, we have uploaded the entire training dataset on [Huggingface](https://huggingface.co/datasets/Michael4933/MGrounding-630k) and organized these massive data collections according to their task class. 
> [!NOTE]
> Due to the nature of multi-image tasks, each training example involves multiple images. As a result, the 600k+ training examples collectively include an even larger number of images.
>
> Please ensure that you have sufficient hard disk storage and a stable internet connection.

You can download the data at `./data/MGrounding-630k` and then simply unzip the corresponding .zip files. This brings you the data structure shown below. We gather all the conversation data at `./data/MGrounding-630k/MGrounding-630k.json` for convenient use, where each training example is labeled with its corresponding sub-task class. The seperate json files for each task is also provided along the way. We just want the best for ya~~~🥰

The downloading code from huggingface is provided in `./data/download.py`, which realizes one-hit quick download.

The final code structure is show as follows:
```
Migician/
├──data/
│  ├──MGrounding-630k
│  │        ├── Common_Object
│  │        │            ├── COCO
│  │        │            ├── ImageNet
│  │        │            ├── Object365
│  │        │            ├── common_train_70k.json # the addtional .zip files at this level may be of limited help
│  │        │
│  │        ├── Difference
│  │        │            ├── clevr-change
│  │        │            ├── img-diff
│  │        │            ├── magicbrush
│  │        │            ├── spot-the-diff
│  │        │            ├── diff_train_70k.json
│  │        │
│  │        ├── Free-Form
│  │        │            ├── Object365
│  │        │            ├── free_form_grounding_130k.json
│  │        │
│  │        ├── Group_Grounding
│  │        │            ├── SA-1B
│  │        │            ├── _gg_reg_40k.json # group grounding reg task
│  │        │            ├── gg_train_120k.json # group grounding rec task
│  │        │
│  │        ├── Object_Tracking
│  │        │            ├── GOT-10k
│  │        │            ├── LaSOT
│  │        │            ├── MOT17_image
│  │        │            ├── TrackingNet
│  │        │            ├── ot_train_130k.json
│  │        │
│  │        ├── Referring_Grounding
│  │        │            ├── ImageNet
│  │        │            ├── refer_train_70k.json
│  │        │
│  │        ├── Region_Locating
│  │                     ├── Object365
│  │                     ├── region_train_70k.json
│  │
│  ├── MGrounding-630k.json # containing all conversation data
│
...
```
An example structure for training data:
```
{
        "id": "5229016_8929009_6793119_3571391", # you can ignore this
        "images": [
            "./MGrounding-630k/Group_Grounding/SA-1B/sa_5229016.jpg",
            "./MGrounding-630k/Group_Grounding/SA-1B/sa_8929009.jpg",
            "./MGrounding-630k/Group_Grounding/SA-1B/sa_6793119.jpg",
            "./MGrounding-630k/Group_Grounding/SA-1B/sa_3571391.jpg"
        ], # they are all organized in the form of a list
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<image>\n<image>\n<image>\nGive the bounding box of the region this sentence refers to: <|object_ref_start|>a statue of a man<|object_ref_end|>." # we adopt special tokens for grounding task
            },
            {
                "from": "gpt",
                "value": "It's in the third image. <|box_start|>(316,58),(764,999)<|box_end|>" # 0-1000, relative position, x1 y1 x2 y2 format
            },
            {
                "from": "human",
                "value": "Recognize the target region that this sentence refers to: <|object_ref_start|>a woman wearing an orange shirt<|object_ref_end|>."
            },
            {
                "from": "gpt",
                "value": "It's in the first image. <|box_start|>(408,656),(578,997)<|box_end|>"
            }
        ],
        "type": "gg_train" # group_grounding task
    }
```

<span id='Inference and Evaluation'/>

### 3. Inference and Evaluation <a href='#all_catelogue'>[Back to Top]</a>

<span id='Inference'/>

#### Inference
As mentioned in the paper, 🎩🎩🎩Migician is finetuned on [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) through a progressive two-stage training process with massive amount of data on 8*A100-80G. You can feel the 🪄🪄🪄magic of multi-image grounding through the following code.

<p align="center">
<img src="figs/multi_view_all.png" width=100%>
</p>

```
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Your_Migician_Path",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # Enabling flash_attention_2 for better acceleration and memory saving is recommended.
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image", "image": resize("./figs/multi_view_1.png"),
            },
            {
                "type": "image", "image": resize("./figs/multi_view_2.png"),
            },
            {
                "type": "image", "image": resize("./figs/multi_view_3.png"),
            },
            {
                "type": "image", "image": resize("./figs/multi_view_4.png"),
            },
            {
                "type": "text", "text": "Please recognize <|object_ref_start|>the common person appearing in all these images<|object_ref_end|> and locate this person in all these image."
            }
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt")
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

<span id='Evaluation'/>

#### Evaluation
🤗📜[MIG-Bench](https://huggingface.co/datasets/Michael4933/MIG-Bench) enables the comprehensive evaluation of current MLLM's MIG ability. Your can directly download it from hugggingface and implement your own evaluation. The file structure for evaluation is as follows:
```
Migician/
├──eval/
│  ├── MIG-Bench
│  │            ├── images
│  │            │       ├── common # 10 diverse tasks
│  │            │       ├── correspondence
│  │            │       ├── group_grounding
│  │            │       ...
│  │            ├── MIG_data.json # could be directly used for evaluation
│  │
│  ├── eval_output/
│  ├── others/ # MMIU and MIBench
│  │
│  ├── MIG_bench_cot.py # Executing MIG through single-image CoT framework
│  ├── MIG_bench_eval.py # Executing MIG by direct inference
│  ├── utils.py
│  ├── requirements.txt
│  ├── chat.py
```

Each testing example is formatted as below, which includes the key informantion such as task class label, image paths, question and ground truth.
> [!NOTE]
> The groundtruth coordinates are normalized as float within 0-1, following the `x1 y1 x2 y2` format.
> 
> The numerical numbers are relative positions regarding the width and height of the whole image.
```
{
        "task": "reasoning",
        "images": [
            "./MIG-Bench/images/reasoning/case097_1.png",
            "./MIG-Bench/images/reasoning/case097_2.png"
        ],
        "question": "Which item in Image-2 share the similar feature of Image-1? Find it and locate it in the second image. ",
        "answer": [
            0.418,
            0.391,
            0.595,
            0.546
        ],
        "additional_info": "Which item in Image-2 share the similar feature of Image-1?",
        "need_format": true
    }
```
You can conduct one-hit evaluation for 🤩🤩🤩<strong>SEVEN</strong> different models[[Migician](https://huggingface.co/Michael4933/Migician), [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [InternVL2](https://huggingface.co/OpenGVLab/InternVL2-8B), [MiniCPM-V_2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6), [LLaVA-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf), [mPLUG-Owl3](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-241101), and [Mantis](https://huggingface.co/TIGER-Lab/Mantis-8B-Idefics2)] on MIG-Bench. Simply run the MIG_bench_eval.py script and it will report IOU@0.7, IOU@0.5, IOU@0.3 and ave-iou scores. We further facilitate the evaluation for 🤗[MIBench](https://huggingface.co/datasets/StarBottle/MIBench) and 🤗[MMIU](https://huggingface.co/MMIUBenchmark/MMIU/tree/main) in MIG_bench_eval.py for different models.


<span id='Finetune'/>

### 4. Finetune  <a href='#all_catelogue'>[Back to Top]</a>
Our two-stage training process is conducted mainly based on 🏭🏭🏭[Llamafactory](https://github.com/hiyouga/LLaMA-Factory), where the whole LLM backbone parameters are finetuned.
We provide our training script for these two stages and the requirements.txt file.
```
Migician/
├── train/
│   ├── stage-1_finetune_full.yaml
│   ├── stage-2_finetune_full.yaml
│   ├── requirements.txt
```


## 📝 Citation
```bibtex
@misc{li2025migicianrevealingmagicfreeform,
      title={Migician: Revealing the Magic of Free-Form Multi-Image Grounding in Multimodal Large Language Models}, 
      author={You Li and Heyu Huang and Chi Chen and Kaiyu Huang and Chao Huang and Zonghao Guo and Zhiyuan Liu and Jinan Xu and Yuhua Li and Ruixuan Li and Maosong Sun},
      year={2025},
      url={https://arxiv.org/abs/2501.05767}, 
}
```

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/liyou/opensource_models/qwen2-vl-7b",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # Enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios is recommended.
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": resize("./figs/multi_view_1.png"),
            },
            {
                "type": "image",
                "image": resize("./figs/multi_view_2.png"),
            },
            {
                "type": "image",
                "image": resize("./figs/multi_view_3.png"),
            },
            {
                "type": "image",
                "image": resize("./figs/multi_view_4.png"),
            },
            {
                "type": "text",
                "text": "Please recognize <|object_ref_start|>the common person appearing in all these images<|object_ref_end|> and locate this person in all these image."
            }
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
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

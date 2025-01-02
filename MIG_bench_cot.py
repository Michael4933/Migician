import torch
import warnings
import torch
import json
import datetime
import random
import copy
from tqdm import tqdm
import torch.multiprocessing as mp
from queue import Empty
from PIL import Image
import string
from decord import VideoReader, cpu 
import warnings

from utils import *
### required by Migician ###
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, LlamaTokenizer

from transformers.image_utils import load_image

### required by LLaVA-OneVision ###
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates, SeparatorStyle
except:
    print('[WARNING] LLaVA hasn\'t been installed')
    
from modelscope import AutoTokenizer, AutoConfig, AutoModel
import warnings

warnings.filterwarnings("ignore")

PROMPT_TEMPLATE = {
    'common_object_1': 'These images share one object in common. Recognize it and tell me its name in single phrase or words.', # single image COT放在ablation study，仅展示个别任务
    'common_object_2': 'Please locate and ground the target object according to the reference:<|object_ref_start|>[RESPONCE]<|object_ref_end|>.',
    'correspondence_1': 'For the first image, briefly describe the semantic/functional feature of the area marked by the red bounding box with really simple words.', # use single phrase to 
    'correspondence_2': 'Ground the area that shares the same semantic or functional meaning of: <ref>[RESPONCE]</ref>.',
    'diff_1': 'Compare these two images carefully and tell me where does they differ. Please answer briefly in single phrase or words.',
    'diff_2': 'According to the object difference/change: [RESPONCE], please ground this difference with bounding box coordinates.',
    'multi_view_1': 'Describe the object in the first image marked with red bounding box with simple phrase or word. You can refer to other images for more precise recognition and description.',
    'multi_view_2': 'Locate and ground the object <|object_ref_start|>[RESPONCE]<|object_ref_end|> with bounding box coordinates.',
    'reasoning_1': ' Name this object in the Image-2 with simple phrase.', #  Please briefly answer in single phrase or words.
    'reasoning_2': 'Please locate and ground the object <|object_ref_start|>[RESPONCE]<|object_ref_end|> with bounding box coordinates.',
    'refer_grounding_1': 'Watch carefully and briefly describe the object in the Image-1 with simple phrase.', # You can refer to Image-2 for more precise recognition and description.
    'refer_grounding_2': 'Please find and ground the object <|object_ref_start|>[RESPONCE]<|object_ref_end|> with bounding box coordinates.',
    'region_1': 'Describe the content of the XXXth picture with simple phrase or words.',
    'region_2': 'Please ground the object <|object_ref_start|>[RESPONCE]<|object_ref_end|> with bounding box coordinates.',
    'view_diff_1': 'Compare these two images carefully and describe the prominent different object with !really simple words or phrase!',
    'view_diff_2': 'Now ground the object difference <ref>[RESPONCE]</ref> with bounding box coordinates.',
    'object_tracking_1': 'Describe the object in the first image marked with red bounding box with simple phrase.',
    'object_tracking_2': 'Now ground the target moving object <ref>[RESPONCE]<ref> with bounding box coordinates.',
    'group_grounding_1': ' Just recognize and tell me which image is it in. Strictly answer from choice list: ',
    'group_grounding_2': '',
    'format_qwen': ' Format:<|box_start|>(x1,y1),(x2,y2)<|box_end|>. Don\'t generate additional words.',
    'format_intern': ' Format:<box>[[x1,y1,x2,y2]]</box>. Don\'t generate additional words.'
}

def compute_iou(ground_truth, prediction, acc, task):
    iou = calculate_iou(ground_truth, prediction)
    if not acc.get(task):
        acc[task]={'IOU@0.7':0,'IOU@0.5':0,'IOU@0.3':0,'AVE_IOU':0,'TOTAL':0}
    if iou >= 0.7: acc[task]['IOU@0.7'] += 1
    if iou >= 0.5: acc[task]['IOU@0.5'] += 1
    if iou >= 0.3: acc[task]['IOU@0.3'] += 1
    acc[task]['AVE_IOU'] += iou
    acc[task]['TOTAL'] += 1
    return iou, acc

def replace_image(match):
    replace_image.counter += 1
    return f"Image-{replace_image.counter}"

def post_processing(acc, output_path, output):
    for task, obj in acc.items():
        print(f"✅ Results for [{task}]:")
        print(obj)
        for item, value in obj.items():
            if item != 'TOTAL':
                print(f"|——> {item}:{100*value/acc[task]['TOTAL']}%  ✨")
    output.append(acc)
    with open(output_path,'w') as file2:
        json.dump(output, file2, indent=4, ensure_ascii=False)

def qwen2_process(obj):
    messages = [{"role": "user","content": []}]
    for path in obj['images']:
        messages[0]["content"].append({"type": "image","image": path})
    
    if obj['need_format']==True:
        obj['question'] += PROMPT_TEMPLATE['format_qwen']
    messages[0]["content"].append({"type": "text", "text": obj['question']})
    return messages

def qwen2_respond(messages, model, processor, device):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,return_tensors="pt")
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response
    
def qwen2_single(model, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"qwen_single_cot_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        task = test_data[0]['task']
        
        # CoT Process: Step 1
        if task == 'reasoning':
            question = obj['question'].replace('Find it and locate it in the second image. ','') + PROMPT_TEMPLATE['reasoning_1']
        elif task == 'group_grounding':
            choice_list = ' | '.join([f'Image{item}' for item in list(range(1,len(obj['images'])+1))])+'.'
            question = obj['question'] + PROMPT_TEMPLATE['group_grounding_1'] + choice_list
        elif task == 'region':
            image_index = get_image_index(obj['question'])+2
            question = PROMPT_TEMPLATE[f'{task}_1'].replace('XXXth',f'{image_index}th')
        else:
            question = PROMPT_TEMPLATE[f'{task}_1']

        messages = qwen2_process(obj)
        response = qwen2_respond(messages, model, processor, device)
        response = response[0].strip().strip('.')
        # print(f'Step-1:\nQuestion: {question}\nResponse: {response}')
        
        # CoT Process: Step 2
        if task in ['correspondence', 'diff', 'reasoning', 'refer_grounding', 'view_diff']: # only require single image
            question = PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_qwen']
            image = obj['images'][1]
        elif task == 'region':
            question = PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_qwen']
            image = obj['images'][0]
        elif task in ['multi_view', 'object_tracking', 'common_object']: # select the target single image
            image_index = get_image_index(obj['question'].replace('first',''))
            assert image_index != -1
            image = obj['images'][image_index]
            question = PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_qwen']
        elif task == 'group_grounding':
            image_index = get_image_index(response)
            if image_index==-1: 
                image_index = 0
            image = obj['images'][image_index]
            question = obj['question']+PROMPT_TEMPLATE['format_qwen']
        
        obj['images'] = [image]
        obj['question'] = question
        messages = qwen2_process(obj)
        response = qwen2_respond(messages, model, processor, device)
        response = response[0].replace('-','').replace('x','')
        # print(f'Step-2:\nQuestion: {question}\nResponse: {response}')
        prediction = extract_bbox(response)  # 正则匹配生成的bbox
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        # print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task': obj['task'], 'question': obj['question'], 'answer': response[0], 'filter_answer': prediction, 'iou': iou, 'groundtruth': obj['answer']})
            
    post_processing(acc, output_path, output)

def qwen2_multi(model, test_data, processor, device, output_path):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"qwen_multi_cot_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        task = test_data[0]['task']
        
        # CoT Process: Step 1
        if task == 'reasoning':
            question = obj['question'].replace('Find it and locate it in the second image. ','') + PROMPT_TEMPLATE['reasoning_1']
        elif task == 'group_grounding':
            choice_list = ' | '.join([f'Image{item}' for item in list(range(1,len(obj['images'])+1))])+'.'
            question = obj['question'] + PROMPT_TEMPLATE['group_grounding_1'] + choice_list
        elif task == 'region':
            image_index = get_image_index(obj['question'])+2
            question = PROMPT_TEMPLATE[f'{task}_1'].replace('XXXth',f'{image_index}th')
        else:
            question = PROMPT_TEMPLATE[f'{task}_1']

        messages = qwen2_process(obj)
        response = qwen2_respond(messages, model, processor, device)
        response = response[0].strip().strip('.')
        # print(f'Step-1:\nQuestion: {question}\nResponse: {response}')
        
        # CoT Process: Step 2
        if task in ['correspondence', 'diff', 'reasoning', 'refer_grounding', 'view_diff']: # only require single image
            question = 'Now ONIY look at Image2. '+PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_qwen']
        elif task == 'region':
            question = 'Now ONIY look at Image1. '+PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_qwen']
        elif task in ['multi_view', 'object_tracking', 'common_object']: # select the target single image
            image_index = get_image_index(obj['question'].replace('first',''))
            # print(obj['question'])
            # print(image_index)
            assert image_index != -1
            question = f"Now ONIY look at the {['first','second','third','fourth','fifth','sixth'][image_index]} image. "+PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_qwen']
        elif task == 'group_grounding':
            image_index = get_image_index(response)
            if image_index==-1: 
                image_index = 0
            question = f"Now ONIY look at the {['first','second','third','fourth','fifth','sixth'][image_index]} image. "+obj['question']+PROMPT_TEMPLATE['format_qwen']
        
        messages = qwen2_process(obj)
        response = qwen2_respond(messages, model, processor, device)
        response = response[0]
        # print(f'Step-2:\nQuestion: {question}\nResponse: {response}')
        prediction = extract_bbox(response)  # 正则匹配生成的bbox
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        # print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task': obj['task'], 'question': obj['question'], 'answer': response[0], 'filter_answer': prediction, 'iou': iou, 'groundtruth': obj['answer']})
            
    post_processing(acc, output_path, output)

def mplug_process(images_all, question, config, tokenizer, processor, device):
    prefix, images = '', []
    for idx, image_path in enumerate(images_all):
        image = Image.open(image_path).convert('RGB')
        prefix += f'Image-{idx+1}:<|image|>\n'
        images.append(image)
    
    question = prefix + question.replace('<image>','')
    question = question.replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>').lstrip()
    question = re.sub(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', r'[\1,\2,\3,\4]', question)
    
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""}
    ]
    inputs = processor(messages, images=images, videos=None).to(device)
    inputs.update({
        'tokenizer': tokenizer,
        'max_new_tokens': 256,
        'decode_text': True,
    })
    
    return inputs

def mplug_single(model, tokenizer, config, processor, test_data, device, output_path):
    model.to(device)
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"mplug_single_cot_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        task = test_data[0]['task']
        
        # CoT Process: Step 1
        if task == 'reasoning':
            question = obj['question'].replace('Find it and locate it in the second image. ','') + PROMPT_TEMPLATE['reasoning_1']
        elif task == 'group_grounding':
            choice_list = ' | '.join([f'Image{item}' for item in list(range(1,len(obj['images'])+1))])+'.'
            question = obj['question'] + PROMPT_TEMPLATE['group_grounding_1'] + choice_list
        elif task == 'region':
            image_index = get_image_index(obj['question'])+2
            question = PROMPT_TEMPLATE[f'{task}_1'].replace('XXXth',f'{image_index}th')
        else:
            question = PROMPT_TEMPLATE[f'{task}_1']

        inputs = mplug_process(obj['images'], question, config, tokenizer, processor, device)
        response = model.generate(**inputs)
        response = response[0].strip().strip('.')
        # print(f'Step-1:\nQuestion: {question}\nResponse: {response}')
        
        # CoT Process: Step 2
        if task in ['correspondence', 'diff', 'reasoning', 'refer_grounding', 'view_diff']: # only require single image
            question = PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
            image = obj['images'][1]
        elif task == 'region':
            question = PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
            image = obj['images'][0]
        elif task in ['multi_view', 'object_tracking', 'common_object']: # select the target single image
            image_index = get_image_index(obj['question'].replace('first',''))
            assert image_index != -1
            image = obj['images'][image_index]
            question = PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
        elif task == 'group_grounding':
            image_index = get_image_index(response)
            if image_index==-1: 
                image_index = 0
            image = obj['images'][image_index]
            question = obj['question']+PROMPT_TEMPLATE['format_intern']
        
        assert isinstance(image, str) # make sure it's single image
        inputs = mplug_process([image], question, config, tokenizer, processor, device)
        response = model.generate(**inputs)
        response = response[0]
        # print(f'Step-2:\nQuestion: {question}\nResponse: {response}')
        prediction = extract_bbox(response)  # 正则匹配生成的bbox
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        # print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task': obj['task'], 'question': obj['question'], 'answer': response[0], 'filter_answer': prediction, 'iou': iou, 'groundtruth': obj['answer']})
            
    post_processing(acc, output_path, output)

def mplug_multi(model, tokenizer, config, processor, test_data, device, output_path):
    model.to(device)
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"mplug_multi_cot_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        task = test_data[0]['task']
        
        # CoT Process: Step 1
        if task == 'reasoning':
            question = obj['question'].replace('Find it and locate it in the second image. ','') + PROMPT_TEMPLATE['reasoning_1']
        elif task == 'group_grounding':
            choice_list = ' | '.join([f'Image{item}' for item in list(range(1,len(obj['images'])+1))])+'.'
            question = obj['question'] + PROMPT_TEMPLATE['group_grounding_1'] + choice_list
        elif task == 'region':
            image_index = get_image_index(obj['question'])+2
            question = PROMPT_TEMPLATE[f'{task}_1'].replace('XXXth',f'{image_index}th')
        else:
            question = PROMPT_TEMPLATE[f'{task}_1']

        inputs = mplug_process(obj['images'], question, config, tokenizer, processor, device)
        response = model.generate(**inputs)
        response = response[0].strip().strip('.')
        # print(f'Step-1:\nQuestion: {question}\nResponse: {response}')
        
        # CoT Process: Step 2
        if task in ['correspondence', 'diff', 'reasoning', 'refer_grounding', 'view_diff']: # only require single image
            question = 'Now ONIY look at Image2. '+PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
        elif task == 'region':
            question = 'Now ONIY look at Image1. '+PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
        elif task in ['multi_view', 'object_tracking', 'common_object']: # select the target single image
            image_index = get_image_index(obj['question'].replace('first',''))
            # print(obj['question'])
            # print(image_index)
            assert image_index != -1
            question = f"Now ONIY look at the {['first','second','third','fourth','fifth','sixth'][image_index]} image. "+PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
        elif task == 'group_grounding':
            image_index = get_image_index(response)
            if image_index==-1: 
                image_index = 0
            question = f"Now ONIY look at the {['first','second','third','fourth','fifth','sixth'][image_index]} image. "+obj['question']+PROMPT_TEMPLATE['format_intern']
        
        inputs = mplug_process(obj['images'], question, config, tokenizer, processor, device)
        response = model.generate(**inputs)
        response = response[0].replace('-','').replace('x','')
        # print(f'Step-2:\nQuestion: {question}\nResponse: {response}')
        prediction = extract_bbox(response)  # 正则匹配生成的bbox
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        # print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task': obj['task'], 'question': obj['question'], 'answer': response[0], 'filter_answer': prediction, 'iou': iou, 'groundtruth': obj['answer']})
            
    post_processing(acc, output_path, output)

def internvl2_process(images, question=None):
    prefix, num_patches_list, first = '', [], True
    # print(images)
    for idx, image_path in enumerate(images):
        temp = load_image_intenvl2(image_path, max_num=12).to(torch.bfloat16).cuda()
        num_patches_list.append(temp.size(0))
        prefix += f'Image-{idx+1}: <image>\n'
        if first == True:
            pixel_values = temp
        else:
            pixel_values = torch.cat((pixel_values, temp), dim=0)
        first = False
    
    question = prefix + question.replace('<image>','')
    question = question.replace('<|box_start|>','<box>').replace('<|box_end|>','</box>').replace('<|object_ref_start|>','<ref>').replace('<|object_ref_end|>','</ref>').lstrip()
    question = re.sub(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', r'[\1,\2,\3,\4]', question)
    return pixel_values, question, num_patches_list

def internvl2_single(model, tokenizer, generation_config, test_data, device, output_path):
    model.to(device)
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_path = os.path.join(output_path, f"internvl2_single_cot_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        task = test_data[0]['task']
        
        # CoT Process: Step 1
        if task == 'reasoning':
            question = obj['question'] + PROMPT_TEMPLATE['reasoning_1']
        elif task == 'group_grounding':
            choice_list = ' | '.join([f'Image{item}' for item in list(range(1,len(obj['images'])+1))])+'.'
            question = obj['question'] + PROMPT_TEMPLATE['group_grounding_1'] + choice_list
        elif task == 'region':
            image_index = get_image_index(obj['question'])+2
            question = PROMPT_TEMPLATE[f'{task}_1'].replace('XXXth',f'{image_index}th')
        else:
            question = PROMPT_TEMPLATE[f'{task}_1']

        pixel_values, question, num_patches_list = internvl2_process(obj['images'], question)
        pixel_values = pixel_values.to(device)
        response, _ = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        response = response.strip().strip('.')
        # print(f'Step-1:\nQuestion: {question}\nResponse: {response}')
        
        # CoT Process: Step 2
        if task in ['correspondence', 'diff', 'reasoning', 'refer_grounding', 'view_diff']: # only require single image
            question = PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
            image = obj['images'][1]
        elif task == 'region':
            question = PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
            image = obj['images'][0]
        elif task in ['multi_view', 'object_tracking', 'common_object']: # select the target single image
            image_index = get_image_index(obj['question'].replace('first',''))
            # print(obj['question'])
            # print(image_index)
            assert image_index != -1
            image = obj['images'][image_index]
            question = PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
        elif task == 'group_grounding':
            image_index = get_image_index(response)
            if image_index==-1: 
                image_index = 0
            image = obj['images'][image_index]
            question = obj['question']+PROMPT_TEMPLATE['format_intern']
        
        assert isinstance(image, str) # make sure it's single image
        pixel_values, question, num_patches_list = internvl2_process([image], question)
        pixel_values = pixel_values.to(device)
        response, _ = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        response = response.replace('-','').replace('x','')
        # print(f'Step-2:\nQuestion: {question}\nResponse: {response}')
        prediction = extract_bbox(response)
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        # print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task': obj['task'], 'question': obj['question'], 'answer': response[0], 'filter_answer': prediction, 'iou': iou, 'groundtruth': obj['answer']})
            
    post_processing(acc, output_path, output)

def internvl2_multi(model, tokenizer, generation_config, test_data, device, output_path):
    model.to(device)
    current_time = datetime.datetime.now().strftime("%m%d_%H%M")
    output_path = os.path.join(output_path, f"internvl_multi_cot_{test_data[0]['task']}_{current_time}.json")
    output, acc = [], {}

    for obj in tqdm(test_data, desc=f"Evaluating {test_data[0]['task']}"):
        task = test_data[0]['task']
        
        # CoT Process: Step 1
        if task == 'reasoning':
            question = obj['question'].replace('Find it and locate it in the second image. ','') + PROMPT_TEMPLATE['reasoning_1']
        elif task == 'group_grounding':
            choice_list = ' | '.join([f'Image{item}' for item in list(range(1,len(obj['images'])+1))])+'.'
            question = obj['question'] + PROMPT_TEMPLATE['group_grounding_1'] + choice_list
        elif task == 'region':
            image_index = get_image_index(obj['question'])+2
            question = PROMPT_TEMPLATE[f'{task}_1'].replace('XXXth',f'{image_index}th')
        else:
            question = PROMPT_TEMPLATE[f'{task}_1']
            
        pixel_values, question, num_patches_list = internvl2_process(obj['images'], question)
        pixel_values = pixel_values.to(device)
        response, history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        response = response.strip().strip('.').replace('-','').replace('x','')
        # print(f'Step-1:\nQuestion: {question}\nResponse: {response}')
        
        # CoT Process: Step 2
        if task in ['correspondence', 'diff', 'reasoning', 'refer_grounding', 'view_diff']: # only require single image
            question = 'Now ONIY look at Image2. '+PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
        elif task == 'region':
            question = 'Now ONIY look at Image1. '+PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
        elif task in ['multi_view', 'object_tracking', 'common_object']: # select the target single image
            image_index = get_image_index(obj['question'].replace('first',''))
            assert image_index != -1
            question = f"Now ONIY look at the {['first','second','third','fourth','fifth'][image_index]} image. "+PROMPT_TEMPLATE[f'{task}_2'].replace('[RESPONCE]',response)+PROMPT_TEMPLATE['format_intern']
        elif task == 'group_grounding':
            image_index = get_image_index(response)
            if image_index==-1: 
                image_index = 0
            question = f"Now ONIY look at the {['first','second','third','fourth','fifth'][image_index]} image. "+obj['question']+PROMPT_TEMPLATE['format_intern']
            
        image = obj['images']
        pixel_values, question, num_patches_list = internvl2_process(image, question)
        pixel_values = pixel_values.to(device)
        
        response = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=history, return_history=False)
        response = response.replace('-','').replace('x','')
        # print(f'Step-2:\nQuestion: {question}\nResponse: {response}')
        prediction = extract_bbox(response)
        iou, acc = compute_iou(obj['answer'], prediction, acc, obj['task'])
        # print(f"GroundTruth: {obj['answer']} | Prediction: {prediction} ——> {iou:.4f}")
        output.append({'task': obj['task'], 'question': obj['question'], 'answer': response[0], 'filter_answer': prediction, 'iou': iou, 'groundtruth': obj['answer']})
            
    post_processing(acc, output_path, output)
    
def model_selection(model_type, model_path, test_data, device, output_path):
    if model_type=='qwen2_vl':
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(device)
        processor = AutoProcessor.from_pretrained(model_path)
        # qwen2_multi(model, test_data, processor, device, output_path)
        qwen2_single(model, test_data, processor, device, output_path)
        
    elif model_type=='internvl2_8b':
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        generation_config = dict(max_new_tokens=1024, do_sample=True)
        # internvl2_multi(model, tokenizer, generation_config, test_data, device, output_path)
        internvl2_single(model, tokenizer, generation_config, test_data, device, output_path)
    
    elif model_type=='mplug_owl3':
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = model.init_processor(tokenizer)
        # mplug_multi(model, tokenizer, config, processor, test_data, device, output_path)
        mplug_single(model, tokenizer, config, processor, test_data, device, output_path)
        
    else:
        raise NotImplementedError

def common_object(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'common_object']
    model_selection(model_type, model_path, test_data, device, output_path)

def diff(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'diff']
    model_selection(model_type, model_path, test_data, device, output_path)

def group_grounding(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'group_grounding']
    model_selection(model_type, model_path, test_data, device, output_path)

def multi_view(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'multi_view']
    model_selection(model_type, model_path, test_data, device, output_path)

def object_tracking(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'object_tracking']
    model_selection(model_type, model_path, test_data, device, output_path)

def view_diff(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'view_diff']
    model_selection(model_type, model_path, test_data, device, output_path)

def refer_grounding(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'refer_grounding']
    model_selection(model_type, model_path, test_data, device, output_path)

def region(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'region']
    model_selection(model_type, model_path, test_data, device, output_path)

def correspondence(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'correspondence']
    model_selection(model_type, model_path, test_data, device, output_path)

def reasoning(model_type, model_path, test_data, device, output_path):
    test_data = [item for item in test_data if item['task'] == 'reasoning']
    model_selection(model_type, model_path, test_data, device, output_path)

def worker(gpu_id, task_queue, model_type, test_data, model_path, output_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f"cuda:{str(gpu_id)}")
    
    while True:
        try:
            task = task_queue.get(timeout=5)
        except Empty:
            break
        task(model_type, model_path, test_data, device, output_path)
        task_queue.task_done()
        

if __name__ == "__main__":
    
    mp.set_start_method("spawn")
    
    with open('./MIG-Bench/MIG_data.json') as file:
        test_data = json.load(file)
        
    output_path = f'./eval_output' # log information output
    os.makedirs(output_path, exist_ok=True)

    ############################################## supported models ###############################################
    ### qwen2_vl, internvl2_8b, mplug_owl3(transformers==4.37.2)
    ### you can modify cot type in model_selection function❤️❤️❤️
    num_gpus, model_type = 4, 'qwen2_vl'
    model_path = "YOUR_PATH"
    print(f'😊 In case you forget~ The model you are evaluating is: {model_type} ❤️\n📂 From path: {model_path}')
    ###############################################################################################################
    
    ############################################### supported tasks ###############################################
    ### MIG-Bench ###
    ### MIG_all, common_object, view_diff, correspondence, diff, group_grounding
    ### multi_view, region, refer_grounding, object_tracking, reasoning
    tasks = [common_object, view_diff, correspondence, diff, group_grounding, multi_view, region, refer_grounding, object_tracking, reasoning]
    ###############################################################################################################
    
    processes = []
    print(model_path)
    
    task_queue = mp.JoinableQueue()
    for task in tasks:
        task_queue.put(task)

    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, task_queue, model_type, test_data, model_path, output_path))
        processes.append(p)
        p.start()

    task_queue.join()
    for p in processes:
        p.join()
    
    print(f'😊 In case you forget~ The model you are evaluating is: {model_type} ❤️\n📂 From path: {model_path}')
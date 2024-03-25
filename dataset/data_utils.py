from typing import List, Optional, Dict, Any, Tuple
import torch
import copy
from .conversation import Conversation, SeparatorStyle, get_conv_template
from .configs import NEW_SPECIAL_TOKENS, PLACE_HOLDERS
from transformers import AutoTokenizer

IGNORE_INDEX = -100


def resized_box(orig_w: int, orig_h: int, resized_w: int, resized_h:int , orig_box: Tuple[int]):
    """
    orig_w, orig_h 는 원래의 너비 높이
    resized_w, resized_h 는 resized 된 영상의 너비 높이
    학습 시 orig_box 는 0~1 로 normalization 된 상태의 xyxy box
    """
    w_ratio = resized_w / orig_w
    h_ratio = resized_h / orig_h
    x1, y1, x2, y2 = orig_box

    new_x1 = x1 * w_ratio
    new_y1 = y1 * h_ratio
    new_x2 = x2 * w_ratio
    new_y2 = y2 * h_ratio 
    new_box = new_x1, new_y1, new_x2, new_y2
    return list(new_box)

def build_conv(source: List[Dict[str, Any]]) -> Conversation:
    conv = get_conv_template('vicuna_v1.1')
    role_map = {"human": conv.roles[0], "gpt": conv.roles[1]}
    assert len(source) > 0
    assert source[0]['from'] == 'human'
    for sentence in source:
        role = role_map[sentence['from']]
        conv.append_message(role, sentence['value'])
    return conv


def normalize_box_xyxy(box: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = box
    x1 /= w
    y1 /= h
    x2 /= w
    y2 /= h
    box = x1, y1, x2, y2
    return box


def denormalize_box_xyxy(box: List[float], w: int, h: int) -> List[float]:
    # Caution. each value can be over than the width or height values.
    x1, y1, x2, y2 = box
    x1 *= w
    y1 *= h
    x2 *= w
    y2 *= h
    box = x1, y1, x2, y2
    return box


def imgph2pattkn(raw_conv, image_token_len: int):
    # This function converts the raw_conv's image placeholder into patch tokens.
    # raw_conv 는 list of dict 자료형태이고, 데이터셋 내 conversations 에 해당하는 부분
    patch_token = NEW_SPECIAL_TOKENS['img_patch']
    replace_token = patch_token * image_token_len
    for sentence in raw_conv:
        sentence['value'] = sentence['value'].replace(PLACE_HOLDERS['image'], replace_token)
    return raw_conv


# 마지막 Preprocessing
def tk_conv_colon_two_train(conv: Conversation, 
                            tokenizer: AutoTokenizer, 
                            max_length: int, 
                            padding: Any, 
                            trigger_id: int = 50262, 
                            system_token_id: int = 50265):
    
    conversation = conv.get_prompt()
    
    input_ids = tokenizer([conversation, ], return_tensors="pt", 
                          max_length=max_length, padding=padding).input_ids[0]
    
    target = copy.deepcopy(input_ids)
    attn_mask = input_ids!=tokenizer.pad_token_id
    pad_length = len(attn_mask) - attn_mask.sum()
    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    context, answer = conversation.split("<ASSISTANT>")
    context_tokens = tokenizer(context)["input_ids"]
    
    instruction_len = len(context_tokens)
    target[~attn_mask] = IGNORE_INDEX
    target[pad_length: pad_length+instruction_len] = IGNORE_INDEX
    
    target[target == system_token_id] = IGNORE_INDEX
    target[-1] = IGNORE_INDEX
    locs = (input_ids == trigger_id).nonzero()

    output_loc_ids = []
    # print(tokenizer.decode(input_ids))
    for loc in locs:
        if pad_length + instruction_len <= loc:
            output_loc_ids.append(loc)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=target,
        all_loc_ids=locs,
        output_loc_ids=output_loc_ids
    )


def ret_preprocess(ret, image_token_len, tokenizer, max_length=512, padding='max_length'):
    # Convert image placeholder into image patch tokens
    # This function can convert only one conversation.
    ret['conversations'] = imgph2pattkn(raw_conv=ret['conversations'], image_token_len=image_token_len)

    # Normalize Bounding Box
    w, h = ret['target']['width'], ret['target']['height']
    normalized_boxes = []
    for bbox in ret['target']['all_boxes']:
        normalized_boxes.append(normalize_box_xyxy(bbox, w, h))
    ret['target']['all_boxes'] = normalized_boxes

    normalized_boxes = []
    for bbox in ret['target']['target_boxes']:
        normalized_boxes.append(normalize_box_xyxy(bbox, w, h))
    ret['target']['target_boxes'] = normalized_boxes
    
    # Convert raw_conv into vicuna conversations
    conv = build_conv(ret['conversations'])
    
    # Tokenize the conversation
    tokenized_conv = tk_conv_colon_two_train(conv=conv, tokenizer=tokenizer, max_length=max_length, padding=padding)
    return ret, tokenized_conv


# https://github.com/huggingface/tokenizers/issues/247#issuecomment-675458087
def setup_for_tokenizer(model, tokenizer, predefined_special_tokens, new_special_tokens):
    """
    special_tokens must be like following
    special_tokens = {
    'bos_token': "<bos>",
    'additional_special_tokens': ["<speaker1>", "<speaker2>"]}
    """
    tokenizer.padding_side = 'left'
    predefined_special_tokens['additional_special_tokens'] = list(new_special_tokens.values())
    num_new_tokens = tokenizer.add_special_tokens(predefined_special_tokens)
    vocab = tokenizer.get_vocab()
    model.resize_token_embeddings(len(vocab))
    
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def get_new_token_ids(tokenizer: AutoTokenizer, new_tokens):
    token_map = {}
    for new_token in new_tokens:
        tkn_id = tokenizer(new_token)
        token_map[new_token] = tkn_id['input_ids'][0]
    return token_map

class Collater:
    def __init__(self, tokenizer: AutoTokenizer, image_token_len: int, image_size:int=384, max_length:int=720):
        self.tokenizer = tokenizer
        self.image_token_len = image_token_len
        self.max_token_length = max_length
        self.image_size = image_size
        self.stack_targets = ['images', 'input_ids', 'labels', 'attention_masks']

    def __call__(self, batch):
        res = {"images": [], 
               "all_boxes": [],
               "target_boxes": [],
               "input_ids": [],
               "attention_masks": [],
               "labels": [],
               "exprs": [],
               "all_loc_ids": [],
               "output_loc_ids": []
               }
               
        for ret in batch:
            r, tkn_conv = ret_preprocess(ret, self.image_token_len, self.tokenizer, self.max_token_length)
            res['images'].append(r['image']) # transformed image to tensors
            
            resized_all_boxes = []
            resized_target_boxes = []

            for box in r['target']['all_boxes']:
                new_box = resized_box(orig_w=r['target']['width'], orig_h=r['target']['height'], 
                                    resized_h=self.image_size, 
                                    resized_w=self.image_size,
                                    orig_box=box)
                resized_all_boxes.append(new_box)

            for box in r['target']['target_boxes']:
                new_box = resized_box(orig_w=r['target']['width'], orig_h=r['target']['height'], 
                                    resized_h=self.image_size, 
                                    resized_w=self.image_size,
                                    orig_box=box)
                
                resized_target_boxes.append(new_box)
            
            res['all_boxes'].append(resized_all_boxes)
            res['target_boxes'].append(resized_target_boxes)
            res['input_ids'].append(tkn_conv["input_ids"])
            res['attention_masks'].append(tkn_conv["attention_mask"])
            res['labels'].append(tkn_conv["labels"])
            res['exprs'].append(r['target']['exprs'])
            res['all_loc_ids'].append(tkn_conv['all_loc_ids']) 
            res['output_loc_ids'].append(tkn_conv['output_loc_ids'])
        
        for item_key in res:
            if item_key in self.stack_targets:
                res[item_key] = torch.stack(res[item_key])
        return res


# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# img_tf = ttf.Compose([ttf.Resize((384, 384)),
#                       ttf.ToTensor(),
#                       ttf.Normalize(mean, std),
#                       ])

# ds1 = RefCOCO("/data/cad-recruit-02_814/kilee/NextChat/data/REC_ref3_train.jsonl", 
#               "/data/cad-recruit-02_814/kilee/NextChat/dataset/template/REC.json",
#               image_transform=img_tf,
#               image_folder="/data/datasets_802/coco/images/train2014")

# model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2')
# tknizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

# setup_for_tokenizer(model, tknizer, PREDEFINED_SPECIAL_TOKENS, NEW_SPECIAL_TOKENS)

# ids = get_new_token_ids(tknizer, list(NEW_SPECIAL_TOKENS.values()))
# print(ids)

# print(tknizer.special_tokens)

# data_collator = Collater(tknizer, 144)

# train_loader = DataLoader(ds1, 2, shuffle=True, collate_fn=data_collator)

# for idx, data in enumerate(train_loader):
#     print(data['labels'].shape)
#     if idx >= 1:
#         break

# output = ret_preprocess(sample_ret1, image_token_len=15, tokenizer=tknizer)
# output2 = ret_preprocess(sample_ret2, image_token_len=15, tokenizer=tknizer)
# input_ids = output[1]['input_ids']
# print(input_ids)
# print(output[1]['attention_mask'])
# indices = (output[1]['attention_mask'] != False).nonzero()
# print(indices)
# print(output[1]['labels'])
# print(tknizer.decode([23998, 25, 220, 50264, 764, 50259]))

# Condition check
# import torch

# a = torch.randn(10)
# b = a <= 0
# indices = b.nonzero()






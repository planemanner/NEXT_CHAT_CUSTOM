import numpy as np
import gradio as gr
from torchvision import transforms
from models.pretrain_config import gpt2_args, swin_args
from models.nextchat import NextChat
from transformers import GPT2Tokenizer
from dataset.data_utils import setup_for_tokenizer, get_new_token_ids, denormalize_box_xyxy
from dataset.configs import PREDEFINED_SPECIAL_TOKENS, NEW_SPECIAL_TOKENS
import torch
import re
from PIL import ImageDraw
from typing import List, Tuple
from misc.data_preprocessing.ms_coco_cat_list import category_map


MAX_TKN_LEN = 512
MAX_NEW_TKN_LEN = 50
PAD_TOKEN_ID = 50257
BOX_TOKEN_ID = 50263
TRIGGER_TOKEN_ID = 50262
IMG_PATCH_TOKEN_ID = 50261
NUM_PATCHES = 144
IMAGE_SIZE = 384
TEXT_COLOR = (0,255,0)
MS_COCO_CAT_LIST = {cat_name: cat_name for cat_name in category_map.values()}

def find_words_in_hash_table(input_string, hash_table):
    words_found = []
    
    # 정규식을 이용하여 입력 문자열에서 단어들을 추출합니다.
    words = re.findall(r'\b\w+\b', input_string)

    # 추출한 단어들을 해시 테이블의 키와 비교하여 값을 찾습니다.
    for word in words:
        if word in hash_table:
            words_found.append(hash_table[word])

    return words_found


def get_ratio(orig_w: int, orig_h: int, cur_w: int, cur_h: int) -> Tuple[float]:
    w_ratio = cur_w / orig_w
    h_ratio = cur_h / orig_h
    return w_ratio, h_ratio

def get_orig_box(resized_box: List[float], w_ratio: float, h_ratio: float, orig_w: int, orig_h: int):
    x1, y1, x2, y2 = resized_box

    x1 *= w_ratio
    x2 *= w_ratio
    y1 *= h_ratio
    y2 *= h_ratio

    x1, x2 = np.clip([x1, x2], a_min=0, a_max=orig_w)
    y1, y2 = np.clip([y1, y2], a_min=0, a_max=orig_h)

    return x1, y1, x2, y2

@torch.no_grad()
def caption_image(image, text):
    """
    :param x: image
    :param text:
    :return:
    """
    caption_input_pattern = re.compile(r'<caption> (?P<boxes>.*?) <end of prefix>')
    is_correct_text = caption_input_pattern.match(text)

    if is_correct_text:
        
        draw = ImageDraw.Draw(image)

        _image = tf(image)
        # input preparation
        image_text = "<im_patch>" * NUM_PATCHES
        query = "<s>" + image_text + "<USER>: <caption> <trigger> <end of prefix>"
        text_chunks = text.split(" ")
        numbers_str = text_chunks[1].strip('[]').split(', ')

        boxes = [float(num) for num in numbers_str]
        _boxes = torch.tensor(boxes, device=device, dtype=torch.float).to(device)
        loc_emb = model.box_enc(_boxes)
        # tokenization and get attention mask
        query_tokens = tokenizer(query, retur_tensors='pt', 
                                 max_length=MAX_TKN_LEN, 
                                 padding='max_length').to(device)
        attn_mask = query_tokens['input_ids'] != PAD_TOKEN_ID

        generated_tokens = model.caption_generate(query_tokens['input_ids'], 
                                                  MAX_NEW_TKN_LEN, 
                                                  loc_emb, 
                                                  do_sample=True,
                                                  attention_mask = attn_mask,
                                                  image=_image)
        
        pred_caption = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        found_mscoco_words = find_words_in_hash_table(pred_caption, MS_COCO_CAT_LIST)
        
        if len(found_mscoco_words) > 0:
            # Do detection when you encounter a coco object name.
            for i in range(len(found_mscoco_words)):
                query = "<s>" + image_text + "<USER>: <detect> <expr> <end_of_prefix>"
                obj_name = found_mscoco_words[i]
                query = query.replace("<expr>", obj_name)

                query_tokens = tokenizer(query, return_tensors='pt', 
                                         max_length=MAX_TKN_LEN, 
                                         padding='max_length').to(device)
                
                generated_tokens, hidden_states = model.detect_generate(query_tokens['input_ids'].to(device),
                                                        max_new_tokens=MAX_NEW_TKN_LEN,
                                                        attention_mask=attn_mask,
                                                        image=_image)
                loc_ids = (generated_tokens == model.trigger_token_id).nonzero()

                bboxes = []

                for loc_id in loc_ids:
                    box_hidden = hidden_states[0, loc_id, :].squeeze(1)
                    bbox = model.box_dec(box_hidden)
                    bboxes.append(bbox.cpu().numpy())

                for box in bboxes:
                    box = np.clip(box, 0, 1)
                    denormalized_box = denormalize_box_xyxy(box, IMAGE_SIZE, IMAGE_SIZE)
                    orig_w, orig_h = image.size()
                    w_ratio, h_ratio = get_ratio(orig_w=orig_w, orig_h=orig_h, cur_w=IMAGE_SIZE, cur_h=IMAGE_SIZE)
                    denormalized_box = get_orig_box(denormalized_box, w_ratio, h_ratio, orig_w=orig_w, orig_h=orig_h)
                    draw.rectangle(denormalized_box, outline=(0,255,0), width = 3)
                    draw.text((denormalized_box[0], denormalized_box[1]), obj_name, TEXT_COLOR)

        denormalized_box = denormalize_box_xyxy(boxes[0], IMAGE_SIZE, IMAGE_SIZE)

        orig_w, orig_h = image.size()

        w_ratio, h_ratio = get_ratio(orig_w=orig_w, orig_h=orig_h, cur_w=IMAGE_SIZE, cur_h=IMAGE_SIZE)

        denormalized_box = get_orig_box(denormalized_box, w_ratio, h_ratio, orig_w=orig_w, orig_h=orig_h)

        draw.rectangle(denormalized_box, outline=(0,255,0), width = 3)

        return image, pred_caption
    else:
        text = "Text 입력 방식이 옳지 않습니다. 입력은 '<caption> [x1, y1, x2, y2] <end of prefix>' 형태여야 합니다."

    return image, text


@torch.no_grad()
def detect_object(image, text):
    """
    :param x:
    :param text:
    :return:
    """
    detect_input_pattern = re.compile(r'<detect> (?P<text>.*?) <end of prefix>')
    is_correct_text = detect_input_pattern.match(text)
    image_text = "<im_patch>" * NUM_PATCHES

    if is_correct_text:
        draw = ImageDraw.Draw(image)

        _image = tf(image)

        query = "<s>" + image_text + "<USER>: <detect> <expr> <end_of_prefix>" 
        query = query.replace("<expr>", text)
        query_tokens = tokenizer(query, return_tensors='pt', 
                                 max_length=MAX_TKN_LEN, 
                                 padding='max_length').to(device)
        attn_mask = query_tokens['input_ids'] != PAD_TOKEN_ID

        generated_tokens, hidden_states = model.detect_generate(query_tokens['input_ids'].to(device),
                                                                max_new_tokens=MAX_NEW_TKN_LEN,
                                                                attention_mask=attn_mask,
                                                                image=_image)
        
        loc_ids = (generated_tokens == model.trigger_token_id).nonzero()

        bboxes = []

        for loc_id in loc_ids:
            box_hidden = hidden_states[0, loc_id, :].squeeze(1)
            bbox = model.box_dec(box_hidden)
            bboxes.append(bbox.cpu().numpy())

        # Draw Box
        for box in bboxes:
            box = np.clip(box, 0, 1)
            denormalized_box = denormalize_box_xyxy(box, IMAGE_SIZE, IMAGE_SIZE)
            orig_w, orig_h = image.size()
            w_ratio, h_ratio = get_ratio(orig_w=orig_w, orig_h=orig_h, cur_w=IMAGE_SIZE, cur_h=IMAGE_SIZE)
            denormalized_box = get_orig_box(denormalized_box, w_ratio, h_ratio, orig_w=orig_w, orig_h=orig_h)
            draw.rectangle(denormalized_box, outline=(0,255,0), width = 3)

        return image
    else:
        text = "Text 입력 방식이 옳지 않습니다. 입력은 '<detect> free-form text <end of prefix>' 형태여야 합니다."
        output_image = image
    # 기 선언된 model 과 tokenizer 를 이용해서 결과물 출력하도록 구현
    return output_image, text


if __name__ == "__main__":
    # 1. model 및 tokenizer load
    device = "cuda:0"
    ckpt_path = "/data/cad-recruit-02_814/kilee/submit/NextChat_epoch_2_iter_20000.pth"
    tkn_path = "/data/cad-recruit-02_814/kilee/NextChat/save_dir/my_tokenizer"
    ckpt = torch.load(ckpt_path)
    # Model initialization
    model = NextChat(gpt2_cfg=gpt2_args, swin_cfg=swin_args)
    tokenizer = GPT2Tokenizer.from_pretrained(tkn_path)
    setup_for_tokenizer(model.lm_decoder, tokenizer, PREDEFINED_SPECIAL_TOKENS, NEW_SPECIAL_TOKENS)
    # Load state dict
    model.load_state_dict(ckpt)
    model.to(device)
    model.set_special_tokens(img_tkn_id=IMG_PATCH_TOKEN_ID, 
                             trigger_tkn_id=TRIGGER_TOKEN_ID, 
                             box_token_id=BOX_TOKEN_ID)
    model.eval()

    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std)])
    
    with gr.Blocks() as demo:
        gr.Markdown("Flip text or image files using this demo.")
        with gr.Tab("Image Captioning"):
            image_for_caption = gr.Image()
            text_for_caption = gr.Textbox()

            # Place holders
            caption_image_place_holder = gr.Image()
            caption_output = gr.Textbox()
            image_caption_button = gr.Button("Caption")

        with gr.Tab("Object Detection"):
            with gr.Row():
                image_for_detection = gr.Image()
                text_for_detection = gr.Textbox()

                # Place holders
                detection_output = gr.Image()

            object_detection_button = gr.Button("Detection")

        # Click 했을 때 액션들
        image_caption_button.click(caption_image,
                                   inputs=[image_for_caption, text_for_caption],
                                   outputs=[caption_image_place_holder, caption_output])

        object_detection_button.click(detect_object,
                                      inputs=[image_for_detection, text_for_detection],
                                      outputs=detection_output)
    demo.launch()

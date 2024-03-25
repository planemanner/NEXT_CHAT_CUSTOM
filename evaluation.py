from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torch.utils.data import DataLoader
import os
import torch
from transformers import AutoTokenizer
from eval_utils import convert2json, get_iou, compute_ap, normalize_confidence
import numpy as np
import argparse
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
from torchvision import transforms
from models.pretrain_config import gpt2_args, swin_args
from models.nextchat import NextChat
from transformers import GPT2Tokenizer
from dataset.data_utils import setup_for_tokenizer, get_new_token_ids, normalize_box_xyxy, resized_box
from dataset.configs import PREDEFINED_SPECIAL_TOKENS, NEW_SPECIAL_TOKENS
from dataset.refcoco import RefCOCO
from tqdm import tqdm
from torch.nn import functional as F


MAX_TKN_LEN = 512
IMAGE_SIZE = 384


def get_caption_scores(ann_file: str, res_file: str):
    """
    item format in res_file
    => List of Dict
    [{"image_id": 404464, "caption": "black and white photo of a man standing in front of a building"}, 
    {"image_id": 380932, "caption": "group of people are on the side of a snowy field"}]

    """
    ann_caption = COCO(ann_file)
    res_caption = ann_caption.loadRes(res_file)
    coco_eval = COCOEvalCap(ann_caption, res_caption)
    coco_eval.params['image_id'] = res_caption.getImgIds()
    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

@torch.no_grad()
def caption_evaluate(model,
                     ann_file: str,
                     val_loader: DataLoader,
                     result_dir: str,
                     tokenizer:AutoTokenizer,
                     device: str,
                     pad_token_id=50257,
                     max_new_tokens=250,
                     num_patches=144):
    
    if not os.path.exists(result_dir):
        print('Caption result directory does not exist')
        os.makedirs(result_dir)

    print('-------Evaluate MS-COCO Captioning--------')
    # evaluation 한 결과를 저장
    result = []

    # Input Query 형태
    # Image, [<caption> token, box loc, <end_of_prefix>]
    # 
    """
    token ids
    "<caption>": 50266,
    "<detect>": 50268,
    "<end of prefix>": 50267,
    """
    image_text = "<im_patch>" * num_patches
    
    for data in tqdm(val_loader):
        query = ["<s> " + image_text + "<USER>: <caption> <trigger> <end of prefix>"]
        images, image_ids = data  # image_ids tensor 형태로 나오니 유념하기.
        images = images.to(device)

        b, c, h, w = images.shape
        query = query * b
        loc = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.float).expand(b, 4)

        loc_emb = model.box_enc(loc)

        query_tokens = tokenizer(query, return_tensors="pt", 
                                 max_length=MAX_TKN_LEN, 
                                 padding='max_length').to(device)
        
        attention_mask = query_tokens['input_ids'] != pad_token_id
        
        generated_tokens = model.caption_generate(query_tokens['input_ids'], 
                                                  max_new_tokens, 
                                                  loc_emb, 
                                                  do_sample=True,
                                                  attention_mask = attention_mask,
                                                  image=images)
        
        # image_save_dir = "/data/cad-recruit-02_814/kilee/NextChat/temporal_results/images"
        for i, (gen_tokens, image_id) in enumerate(zip(generated_tokens, image_ids)):
            pred_caption = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            # print(f'{image_id}')
            # cur_pil_img = val_loader.dataset._load_image(image_id.item())
            # save_path = os.path.join(image_save_dir, f"MSCOCO_{image_id}.png")
            # cur_pil_img.save(save_path)
            # print(pred_caption[1:])
            
            result.append({"image_id": image_id.item(), "caption": pred_caption[1:]})
        
    result_path = convert2json(result, result_dir)
    # 값 저장해서 Logging 해보기
    get_caption_scores(ann_file, result_path)


@torch.no_grad()
def detection_evaluate(model, 
                       val_loader: DataLoader,
                       tokenizer:AutoTokenizer,
                       device: str,
                       max_new_tokens=50,
                       num_patches=144,
                       iou_threshold=0.5,
                       pad_token_id=50257
                       ):
    
    image_text = "<im_patch>" * num_patches
    
    tp, tn, fp, fn = 0, 0, 0, 0
    total = 0

    results = {"tp_or_fp": [], "confidences": []}

    for data in tqdm(val_loader):
        
        text_query = ["<s> " + image_text + " <USER>: <detect> <expr> <end_of_prefix>"]
        images, targets = data
        images = images.to(device)
        exprs = targets['exprs']
        b, c, h, w = images.shape
        text_query = text_query * b
        total += b
        for i, expr in enumerate(exprs):
            text_query[i] = text_query[i].replace("<expr>", expr)
        
        query_tokens = tokenizer(text_query, 
                                 return_tensors="pt", 
                                 max_length=MAX_TKN_LEN, 
                                 padding='max_length').to(device)
        
        attn_mask = query_tokens['input_ids'] != pad_token_id
        
        batch_generated_tokens, hidden_states = model.detect_generate(query_tokens["input_ids"], 
                                                                      attention_mask=attn_mask,
                                                                      max_new_tokens=max_new_tokens, 
                                                                      image=images,
                                                                      pad_token_id=pad_token_id)
        
        for b_id, gen_tokens in enumerate(batch_generated_tokens):
            gt_box = targets['all_bboxes'][b_id]
            loc_ids = (gen_tokens == model.trigger_token_id).nonzero()

            if len(loc_ids) > 0:
                
                box_hiddens = hidden_states[b_id, loc_ids, :].squeeze(1)
                token_logits = model.lm_decoder.lm_head(box_hiddens)
                
                token_probs = F.softmax(token_logits, dim=1)
                objectness = token_probs[:, model.trigger_token_id]
                values, best_idx = torch.max(objectness, dim=0)

                # all_conf.extend(values.cpu().tolist())

                best_box_hidden = hidden_states[b_id, loc_ids[best_idx], :]
                loc_pred = model.box_dec(best_box_hidden).cpu().numpy()
                
                # print(loc_pred)
                iou = get_iou(loc_pred[0], gt_box)

                if iou > iou_threshold:
                    tp += 1
                    results['tp_or_fp'].append(1)
                else:
                    fp += 1
                    results['tp_or_fp'].append(0)

                conf = objectness[best_idx].cpu().numpy().item()
                results["confidences"].append(conf)
                
            else:
                fn += 1
        
        cur_precision = tp / (tp + fp) * 100
        cur_recall = tp / (tp + fn) * 100
        print(f"Precision : {cur_precision}")
        print(f"Recall : {cur_recall}")

    results['confidences'] = normalize_confidence(results['confidences'])

    sorted_indices = sorted(range(len(results['confidences'])), key = lambda k : results['confidences'][k], reverse=True)
    sorted_tp_fp = [results['tp_or_fp'][idx] for idx in sorted_indices]
    
    results['tp_or_fp'] = sorted_tp_fp
    
    ap = compute_ap(results)
    print(f"Average Precision : {ap}")

def detection_collate(batch):
    _targets = batch
    images = []
    targets = {"exprs": [],
               "all_bboxes": []}
    
    for _target in _targets:
        images.append(_target["image"])
        
        normalized_box = normalize_box_xyxy(_target['target']['all_boxes'][0], _target['target']['width'], _target['target']['height'])
        box = resized_box(_target['target']['width'], _target['target']['height'], IMAGE_SIZE, IMAGE_SIZE, normalized_box)
        targets["all_bboxes"].append(box)
        targets["exprs"].append(_target['target']['exprs'])
    
    images = torch.stack(images)

    return images, targets


def evaluation(args):
    device = args.device
    ckpt = torch.load(args.model_ckpt)

    model = NextChat(gpt2_cfg=gpt2_args, swin_cfg=swin_args)
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    setup_for_tokenizer(model.lm_decoder, tokenizer, PREDEFINED_SPECIAL_TOKENS, NEW_SPECIAL_TOKENS)
    sepcial_tokens = tokenizer.additional_special_tokens
    token_map = get_new_token_ids(tokenizer, sepcial_tokens)

    model.set_special_tokens(img_tkn_id=token_map['<im_patch>'],
                             trigger_tkn_id=token_map['<trigger>'],
                             box_token_id=token_map['<boxes>'])
    
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std)])
    
    if args.task == "caption":
        val_cap_dataset = CocoCaptions(root=args.val_img_dir, 
                                       annFile=args.ann_path, 
                                       transform=tf)
        
        val_loader = DataLoader(val_cap_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False, num_workers=8,
                                )
        
        caption_evaluate(model, 
                         ann_file=args.ann_path, 
                         val_loader=val_loader, 
                         result_dir=args.result_dir, 
                         tokenizer=tokenizer,
                         max_new_tokens=args.max_new_tkn, 
                         device=args.device)

    if args.task == "detection":

        val_det_dataset = RefCOCO(filename=args.refcoco_valfile, 
                                  template_file=args.refcoco_template,
                                  image_folder=args.val_img_dir,
                                  image_transform=tf)
        
        val_loader = DataLoader(val_det_dataset,
                                batch_size=args.batch_size,
                                shuffle=False, num_workers=8,
                                collate_fn=detection_collate
                                )
        
        detection_evaluate(model, 
                           val_loader=val_loader, 
                           max_new_tokens=args.max_new_tkn,
                           tokenizer=tokenizer, device=args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, 
                        default="/data/cad-recruit-02_814/kilee/NextChat/save_dir/model/NextChat_epoch_1_iter_140000.pth")
    parser.add_argument("--tokenizer_path", type=str, default="/data/cad-recruit-02_814/kilee/NextChat/save_dir/my_tokenizer")
    parser.add_argument("--val_img_dir", type=str, default="/data/datasets_802/coco/images/val2014")
    parser.add_argument("--ann_path", type=str, default="/data/datasets_802/coco/annotations/captions_val2014.json")
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="caption", choices=["caption", "detection"])
    parser.add_argument("--max_new_tkn", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--refcoco_template", type=str, default="/data/cad-recruit-02_814/kilee/NextChat/dataset/template/REC.json")
    parser.add_argument("--refcoco_valfile", type=str, default="/data/cad-recruit-02_814/kilee/NextChat/data/REC_refcoco+_unc_val.jsonl")
    args = parser.parse_args()
    evaluation(args)
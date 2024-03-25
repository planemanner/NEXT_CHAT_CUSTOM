# import torch

# # 주어진 텐서
# a = torch.Tensor([[0, 1, 2, 3, 2], [2, 3, 4, 5, 2], [3, 4, 2, 6, 2]])

# # 각 행별로 최초의 "2"의 인덱스 찾기
# first_two_indices = (a == 2).nonzero()[:, 1]

# # 결과 출력
# print("각 행별로 최초의 '2'의 인덱스:", first_two_indices)



# import torch
# from torchvision import transforms as ttf
# from dataset.refcoco import RefCOCO
# from transformers import GPT2Tokenizer
# from models.nextchat import NextChat
# from models.pretrain_config import gpt2_args, swin_args
# from dataset.data_utils import setup_for_tokenizer, get_new_token_ids, ret_preprocess
# from dataset.configs import PREDEFINED_SPECIAL_TOKENS, NEW_SPECIAL_TOKENS

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

# model = NextChat(swin_cfg=swin_args, gpt2_cfg=gpt2_args)
# tknizer = GPT2Tokenizer.from_pretrained("/data/cad-recruit-02_814/kilee/NextChat/save_dir/my_tokenizer")

# setup_for_tokenizer(model.lm_decoder, tknizer, PREDEFINED_SPECIAL_TOKENS, NEW_SPECIAL_TOKENS)

# ids = get_new_token_ids(tknizer, list(NEW_SPECIAL_TOKENS.values()))

# sample_ret1 = ds1.__getitem__(0)

# output = ret_preprocess(sample_ret1, image_token_len=15, tokenizer=tknizer)
# # print(output)




# def calculate_iou(box1, box2):
#     # 각 박스의 좌표 추출
#     x1, y1, x2, y2 = box1
#     x3, y3, x4, y4 = box2

#     # 교차하는 부분의 좌표 계산
#     intersection_x1 = max(x1, x3)
#     intersection_y1 = max(y1, y3)
#     intersection_x2 = min(x2, x4)
#     intersection_y2 = min(y2, y4)

#     # 교차하는 부분의 넓이 계산
#     intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)

#     # 각 박스의 넓이 계산
#     box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
#     box2_area = (x4 - x3 + 1) * (y4 - y3 + 1)

#     # 합집합의 넓이 계산
#     union_area = box1_area + box2_area - intersection_area

#     # IoU 계산
#     iou = intersection_area / union_area

#     return iou

# # 주어진 두 바운딩 박스의 IoU 계산
# box1 = [0.0, 0.0, 1, 1]
# box2 = [0.0, 0.0, 1, 1]

# iou = calculate_iou(box1, box2)
# print("IoU:", iou)

import logging
from train_utils import init_distributed_setup, set_random_seed, LinearwarmupCosineScheduler, save_ddp_model, save_tokenizer, get_transform_train
import argparse
import torch
from models.nextchat import NextChat
from torchvision import transforms as ttf
from dataset.concatenate_dataset import ConcatDataset
from dataset.refcoco import RefCOCO
from dataset.refcocog import RefCOCOG
from torch.utils.data.distributed import DistributedSampler
from dataset.data_utils import Collater, setup_for_tokenizer, get_new_token_ids
from models.pretrain_config import gpt2_args, swin_args
from transformers import GPT2Tokenizer
from dataset.configs import PREDEFINED_SPECIAL_TOKENS, NEW_SPECIAL_TOKENS
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from loss_funcs import NextChatLoss
from tqdm import tqdm
from dataset.vg_dataset import VisualGenome, VisualGenomeDet
from dataset.mscoco_caption import MsCaption
from dataset.mscoco_det import MscocoDet
from evaluation import caption_evaluate, detection_evaluate
import os
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco2014_img_dir', type=str, default="/data/datasets_802/coco/images/train2014")
    parser.add_argument('--coco2017_img_dir', type=str, default="/data/datasets_802/coco/images/train2017")
    parser.add_argument('--vg_img_dir', type=str, default="/data/datasets_802/VisualGenome/images")

    parser.add_argument('--vg_det_template', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/dataset/template/VG_det.json")
    parser.add_argument('--vg_cap_template', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/dataset/template/VG_caption.json")

    parser.add_argument('--refcoco_template', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/dataset/template/REC.json")
    parser.add_argument('--refcoco_plus_template', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/dataset/template/REC.json")
    parser.add_argument('--refcocog_template', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/dataset/template/REG.json")
    parser.add_argument('--cococap_template', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/dataset/template/mscoco_caption.json")
    parser.add_argument('--cocodet_template', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/dataset/template/mscoco_detection.json")

    parser.add_argument('--vg_det_data', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/data/vg_det_train.jsonl")
    parser.add_argument('--vg_cap_data', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/data/genome_correct_train.jsonl")
    parser.add_argument('--refcoco_data', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/data/REC_ref3_train.jsonl")
    parser.add_argument('--refcoco_plus_data', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/data/REC_ref3_train.jsonl")
    parser.add_argument('--refcocog_data', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/data/REC_ref3_train.jsonl")

    parser.add_argument('--cococap2014_data', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/data/coco_caption_train2014.jsonl")
    parser.add_argument('--cococap2017_data', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/data/coco_caption_train2017.jsonl")
    parser.add_argument('--cocodet2014_data', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/data/train_coco_det_2014.jsonl")
    parser.add_argument('--cocodet2017_data', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/data/train_coco_det_2017.jsonl")

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--wd', type=float, default=0.0, help="weight decay")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--train_verbose_period', type=int, default=100)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--num_img_patches', type=int, default=144)
    parser.add_argument('--max_tkn_len', type=int, default=512)
    parser.add_argument('--save_eval_period', type=int, default=10000)
    parser.add_argument('--model_save_dir', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/save_dir/model")
    parser.add_argument('--tokenizer_save_dir', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/save_dir/my_tokenizer")
    parser.add_argument('--tensorboard_dir', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/save_dir/training_logs")
    parser.add_argument('--local-rank', type=int, default=0)

    parser.add_argument('--ref3det_val_img_dir', type=str, default="/data/datasets_802/coco/images/train2014")
    parser.add_argument('--refcoco_val', type=str, default="/data/datasets_802/refcoco_ann/annotations/finetune_refcoco_val.json")
    parser.add_argument('--refcocog_val', type=str, default="/data/datasets_802/refcoco_ann/annotations/finetune_refcocog_val.json")
    parser.add_argument('--refcoco_plus_val', type=str, default="/data/datasets_802/refcoco_ann/annotations/finetune_refcoco+_val.json")

    parser.add_argument('--cap2014_val_img_dir', type=str, default="/data/datasets_802/coco/images/val2014")
    parser.add_argument('--cap2017_val_img_dir', type=str, default="/data/datasets_802/coco/images/val2017")
    parser.add_argument('--cap2014_val_ann', type=str, default="/data/datasets_802/coco/annotations/captions_val2014.json")
    parser.add_argument('--cap2017_val_ann', type=str, default="/data/datasets_802/coco/annotations/captions_val2017.json")
    parser.add_argument('--caption_result_dir_2014', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/cap_2014_result")
    parser.add_argument('--caption_result_dir_2017', type=str, default="/data/cad-recruit-02_814/kilee/NextChat/cap_2017_result")
    return parser.parse_args()


def train(args):
    set_random_seed(args.seed)

    init_distributed_setup(args)
    logger = logging.getLogger("training logger")	
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    image_transform = ttf.Compose([ttf.Resize((args.img_size, args.img_size)),
                                   ttf.ToTensor(),
                                   ttf.Normalize(image_mean, image_std)])
    
    logger.info('----load training dataset...-----')
    # vg_det_ds = VisualGenomeDet(filename=args.vg_det_data, template_file=args.vg_det_template, 
    #                             image_transform=image_transform, image_folder=args.vg_img_dir, 
    #                             seed=args.seed)
    
    refcoco_ds = RefCOCO(filename=args.refcoco_data, template_file=args.refcoco_template,
                         image_transform=image_transform, image_folder=args.coco2014_img_dir, seed=args.seed)
    
    ref_plus_ds = RefCOCO(filename=args.refcoco_plus_data, template_file=args.refcoco_template,
                          image_transform=image_transform, image_folder=args.coco2014_img_dir, seed=args.seed)
    
    refg_ds = RefCOCOG(filename=args.refcocog_data, template_file=args.refcocog_template, 
                       image_transform=image_transform, image_folder=args.coco2014_img_dir, seed=args.seed)
    
    vg_cap_ds = VisualGenome(filename=args.vg_cap_data, template_file=args.vg_cap_template, 
                             image_transform=image_transform, image_folder=args.vg_img_dir, 
                             seed=args.seed)
    
    cococap_2014 = MsCaption(filename=args.cococap2014_data, template_file=args.cococap_template, 
                             image_transform=image_transform, image_folder=args.coco2014_img_dir, seed=args.seed)
    
    cococap_2017 = MsCaption(filename=args.cococap2017_data, template_file=args.cococap_template, 
                             image_transform=image_transform, image_folder=args.coco2017_img_dir, seed=args.seed)
    
    cocodet_2014 = MscocoDet(filename=args.cocodet2014_data, template_file=args.cocodet_template, 
                             image_transform=image_transform, image_folder=args.coco2014_img_dir, seed=args.seed)
    
    cocodet_2017 = MscocoDet(filename=args.cocodet2017_data, template_file=args.cocodet_template, 
                             image_transform=image_transform, image_folder=args.coco2017_img_dir, seed=args.seed)
    
    all_data = [vg_cap_ds, refcoco_ds, ref_plus_ds, refg_ds, cococap_2014, cococap_2017, cocodet_2014, cocodet_2017]
    
    train_dataset = ConcatDataset(all_data)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    logger.info('----Completed loading training dataset-----')

    logger.info('----Completed loading validation datasets----')

    logger.info('----Staring model setup...----')
    
    model = NextChat(gpt2_cfg=gpt2_args, swin_cfg=swin_args)

    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_args['pretrain_model'])
    tokenizer.bos_token = PREDEFINED_SPECIAL_TOKENS['bos_token']
    tokenizer.eos_token = PREDEFINED_SPECIAL_TOKENS['eos_token']
    tokenizer.pad_token = PREDEFINED_SPECIAL_TOKENS['pad_token']
    
    setup_for_tokenizer(model.lm_decoder, tokenizer, PREDEFINED_SPECIAL_TOKENS, NEW_SPECIAL_TOKENS)
    new_token_maps = get_new_token_ids(tokenizer=tokenizer, new_tokens=list(NEW_SPECIAL_TOKENS.values()))
    save_tokenizer(tokenizer, args.tokenizer_save_dir)
    
    model.set_special_tokens(img_tkn_id=new_token_maps[NEW_SPECIAL_TOKENS['img_patch']],
                             trigger_tkn_id=new_token_maps[NEW_SPECIAL_TOKENS['trigger']])
    
    logger.info('----Model Setup Done----')

    collate_obj = Collater(tokenizer=tokenizer, 
                           image_token_len=args.num_img_patches,
                           max_length=args.max_tkn_len)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=int(args.batch_size / args.world_size),
                              num_workers=int(args.num_workers / args.world_size),
                              sampler=train_sampler,
                              collate_fn=collate_obj,
                              pin_memory=True
                              )
    
    model = model.cuda(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    criterion = NextChatLoss(vocap_size=len(tokenizer.get_vocab())).to(args.local_rank)

    total_iteration = train_loader.__len__() * args.num_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    lr_scheduler = LinearwarmupCosineScheduler(optimizer, 
                                               total_iterations=total_iteration, 
                                               warmup_ratio=args.warmup_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    global_step = 0
    logger.info("----------Training Start ! -----------------")
    for epoch in range(args.num_epochs):
        # AutoCase 추가 필요
        model.train()

        for i, data in enumerate(tqdm(train_loader)):
            # Not nested form
            images, input_ids, labels, attention_masks = data['images'], data['input_ids'], data['labels'], data['attention_masks']
            # Nested form
            all_boxes, tgt_boxes, all_loc_ids, output_loc_ids = data['all_boxes'], data['target_boxes'], data['all_loc_ids'], data['output_loc_ids']
            images, input_ids, labels, attention_masks = images.to(args.local_rank), input_ids.to(args.local_rank), labels.to(args.local_rank), attention_masks.to(args.local_rank)

            with torch.autocast(device_type="cuda"):
                logits, loc_preds, output_loc_hiddens, encoded_loc_preds, dec_loc_embeds = model(input_ids, images, 
                                                                                                 all_boxes, all_loc_ids, 
                                                                                                 output_loc_ids, attention_masks)

                loss, loss_logs = criterion(logits, labels, loc_preds, all_boxes, tgt_boxes, 
                                            output_loc_hiddens, dec_loc_embeds, encoded_loc_preds,
                                            device=f"cuda:{args.local_rank}")
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            
            with torch.no_grad():
                for loss_name in loss_logs:
                    writer.add_scalar(loss_name, loss_logs[loss_name], global_step)
                global_step += 1

                if (i + 1) % args.train_verbose_period == 0:

                    print(f"EPOCH : {epoch} / {args.num_epochs}, ITERATION : {i+1} / {total_iteration}, TRAIN LOSS : {loss.item()}")

                    for loss_cat, loss_val in loss_logs.items():
                        print(f"{loss_cat} : {loss_val}")
                if (i + 1) % args.save_eval_period == 0:
                    save_path = os.path.join(args.model_save_dir, f"NextChat_epoch_{epoch+1}_iter_{i+1}.pth")
                    save_ddp_model(model, save_path)

    writer.close()             
    logger.info("-----------Finished Training ! ---------------")

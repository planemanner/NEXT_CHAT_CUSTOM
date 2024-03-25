from torch import nn
import torch
from transformers import GPT2LMHeadModel, SwinModel
from .sub_modules import LocEncoder, LocDecoder, MMProjector
from .model_utils import disable_learning


class NextChat(nn.Module):
    def __init__(self, gpt2_cfg, swin_cfg, img_token_id=None, 
                 trigger_token_id=None, im_start_id=None, im_end_id=None,
                 box_token_id=None):
        super(NextChat, self).__init__()
        self.num_patches = swin_cfg['num_patches']
        self.vision_encoder = SwinModel.from_pretrained(swin_cfg['pretrain_model'])
        self.lm_decoder = GPT2LMHeadModel.from_pretrained(gpt2_cfg['pretrain_model'], torch_dtype='auto')

        self.mm_projector = MMProjector(input_dim=swin_cfg['last_hidden_dim'], 
                                        out_dim=gpt2_cfg['last_hidden_dim'])
        
        self.box_enc = LocEncoder(out_dim=gpt2_cfg['last_hidden_dim'])
        self.box_dec = LocDecoder(input_dim=gpt2_cfg['last_hidden_dim'])

        self.img_token_id = img_token_id
        self.trigger_token_id = trigger_token_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.box_token_id = box_token_id
        # Freeze vision encoder
        disable_learning(self.vision_encoder)

    def set_special_tokens(self, img_tkn_id, trigger_tkn_id, box_token_id=None, im_start_id=None, im_end_id=None):
        self.img_token_id = img_tkn_id        
        self.trigger_token_id = trigger_tkn_id
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.box_token_id = box_token_id
        

    def forward(self, input_ids, 
                input_image, 
                bboxes=None, 
                all_loc_ids=None, 
                output_loc_ids=None, 
                attention_masks=None):
        """
        input_image must be processed state by AutoImageProcessor of huggingface transformers
        loc : normalized loc such as [0.1, 0.2, 0.3, 0.4], xyxy format
        input_ids only have prompt by human denoted as user.
        """
        vision_output = self.vision_encoder(input_image)
        visual_feat = vision_output['last_hidden_state']
        proj_vis_tks = self.mm_projector(visual_feat)
        
        emb_tokens = self.lm_decoder.transformer.wte(input_ids).half()
        
        if all_loc_ids is not None:
            loc_embeds = []
            for b_id, loc_ids in enumerate(all_loc_ids):
                if len(loc_ids) > 0:
                    
                    cur_boxes = torch.tensor([bboxes[b_id]], device=emb_tokens.device)
                    cur_loc_emb = self.box_enc(cur_boxes).squeeze(0)
                    emb_tokens[b_id, loc_ids, :] = cur_loc_emb.unsqueeze(1)
                    loc_embeds.append(cur_loc_emb)
            
            loc_embeds = torch.cat(loc_embeds)

        with torch.no_grad():
            vis_masks = input_ids == self.img_token_id
            for b_id, vis_mask in enumerate(vis_masks):
                emb_tokens[b_id, vis_mask, :] = proj_vis_tks[b_id, ...]
        
        outputs = self.lm_decoder(inputs_embeds=emb_tokens, 
                                  attention_mask=attention_masks, 
                                  output_hidden_states=True)
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        loc_preds = [] # For box loss
        output_loc_hiddens = []
        encoded_loc_preds = []

        if output_loc_ids is not None:
            for b_id, loc_ids in enumerate(output_loc_ids):
                if len(loc_ids) > 0:
                    to_be_decoded = hidden_states[b_id, loc_ids, :]
                    output_loc_hiddens.append(to_be_decoded)

            if len(output_loc_hiddens) > 0:
                output_loc_hiddens = torch.cat(output_loc_hiddens)

                loc_preds = self.box_dec(output_loc_hiddens)

                encoded_loc_preds = self.box_enc(loc_preds)
        
        if len(loc_embeds) > 0:
            dec_loc_embeds = self.box_dec(loc_embeds)
        else:
            dec_loc_embeds = None
        return logits, loc_preds, output_loc_hiddens, encoded_loc_preds, dec_loc_embeds

    @torch.no_grad()
    def caption_generate(self, input_ids, 
                         max_new_tokens, 
                         loc_embs=None, 
                         attention_mask=None,
                         image=None, do_sample=False, 
                         top_k=50, top_p=0.5,
                         pad_token_id=50257,
                         num_return_sequences=1,
                         temperature=0.7) -> torch.Tensor:
        # generate is not differentiable.
        vision_output = self.vision_encoder(image)
        visual_feat = vision_output['last_hidden_state']
        proj_vis_tks = self.mm_projector(visual_feat)
        emb_tokens = self.lm_decoder.transformer.wte(input_ids)
        
        vis_masks = input_ids==self.img_token_id
        
        for b_id, vis_mask in enumerate(vis_masks):
            emb_tokens[b_id, vis_mask, :] = proj_vis_tks[b_id, ...]
            
        if loc_embs is not None:
            loc_masks = input_ids == self.trigger_token_id

            for b_id, loc_mask in enumerate(loc_masks):
                emb_tokens[b_id, loc_mask, :] = loc_embs[b_id, ...]
        
        decoded_tokens = self.lm_decoder.generate(attention_mask = attention_mask, 
                                                  inputs_embeds=emb_tokens, 
                                                  max_new_tokens=max_new_tokens,
                                                  do_sample=do_sample,
                                                  pad_token_id=pad_token_id,
                                                  top_k=top_k,
                                                  length_penalty=1.0,
                                                  top_p=top_p,
                                                  eos_token_id=50259,
                                                  temperature=temperature,
                                                  repetition_penalty=2.0,
                                                  no_repeat_ngram_size=3,
                                                  num_return_sequences=num_return_sequences,
                                                  )
        
        return decoded_tokens

    @torch.no_grad()
    def detect_generate(self, input_ids, 
                        max_new_tokens,
                        attention_mask=None,
                        image=None, do_sample=True,
                        top_k=50, top_p=0.5,
                        pad_token_id=50257) -> torch.Tensor:
        # generate is not differentiable.
        
        vision_output = self.vision_encoder(image)
        visual_feat = vision_output['last_hidden_state']
        proj_vis_tks = self.mm_projector(visual_feat)
        emb_tokens = self.lm_decoder.transformer.wte(input_ids)
  
        vis_masks = input_ids==self.img_token_id

        for b_id, vis_mask in enumerate(vis_masks):
            emb_tokens[b_id, vis_mask, :] = proj_vis_tks[b_id, ...]
        
        outputs = self.lm_decoder(inputs_embeds=emb_tokens, output_hidden_states=True)
        
        generated_tokens = self.lm_decoder.generate(
            attention_mask = attention_mask, 
            inputs_embeds=emb_tokens, 
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=pad_token_id,
            temperature=2.5,
            )
            
        
        hidden_states = outputs.hidden_states[-1]

        return generated_tokens, hidden_states
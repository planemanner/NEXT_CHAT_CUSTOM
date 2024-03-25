import torch
from torch import nn
from torchvision.ops.boxes import box_area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # left top
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # right bottom
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area  


class NextChatLoss(nn.Module):
    def __init__(self, vocap_size: int, alpha:float = 2, beta:float = 0.2):
        super(NextChatLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = alpha  # L_{det}'s l1 loss coefficient
        self.beta = beta # L_{det}'s GIoU loss coefficient
        self.vocap_size = vocap_size

    def forward(self, 
                logits, 
                token_targets, 
                loc_preds, 
                all_boxes,
                tgt_boxes,
                output_loc_hiddens,
                dec_loc_embeds,
                encoded_loc_preds,
                device="cuda:0"):
        
        # Initialization
        total_loss = 0
        l_text = 0
        l_det = 0
        l_cyc = 0

        # HugginfFace Implementation format
        if token_targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = token_targets[..., 1:].contiguous()
            # x, y = shift_labels[shift_labels > 29871], shift_logits.argmax(-1)[shift_labels > 29871]

            # Flatten the tokens
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            l_text += self.cross_entropy(shift_logits, shift_labels)

        if tgt_boxes is not None:
            decoder_tgt_boxes = []
            num_loc_preds = len(loc_preds)

            for boxes in tgt_boxes:
                if len(boxes) > 0:
                    decoder_tgt_boxes.append(torch.tensor(boxes, device=device))
            
            if num_loc_preds > 0:
                decoder_tgt_boxes = torch.cat(decoder_tgt_boxes)
                l_det += (self.alpha * self.l1_loss(loc_preds, decoder_tgt_boxes))
                
                is_positive_mask = (loc_preds >= 0).all(dim=1)
                
                giou_mask = (loc_preds[:, 2:] >= loc_preds[:, :2]).all(-1) * is_positive_mask
                
                if giou_mask.sum() > 0:
            
                    giou_loss = 1 - torch.diag(generalized_box_iou(loc_preds[giou_mask, :], decoder_tgt_boxes[giou_mask, :]))
                    l_det += (self.beta * giou_loss.mean())

        # Decoder Cylce
        if num_loc_preds > 0:
            l_cyc += self.l2_loss(encoded_loc_preds, output_loc_hiddens)

        if len(dec_loc_embeds) > 0:
            all_target_boxes = []
            for boxes in all_boxes:
                if len(boxes) > 0:
                    all_target_boxes.append(torch.tensor(boxes, device=device))
            all_target_boxes = torch.cat(all_target_boxes)

            l_cyc += self.l1_loss(dec_loc_embeds, all_target_boxes)

        total_loss = l_text + l_det + l_cyc

        return total_loss, {"total loss": total_loss.item()}


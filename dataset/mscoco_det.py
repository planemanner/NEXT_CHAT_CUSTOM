from .base_dataset import BaseDataset
from .configs import PLACE_HOLDERS, NEW_SPECIAL_TOKENS


class MscocoDet(BaseDataset):
    def __init__(self, filename, 
                 template_file,
                 image_transform=None,
                 image_folder=None, seed=42):
        super().__init__(filename, template_file, image_folder, seed)
        self.image_transform = image_transform
    
    def __getitem__(self, index):
        item = self.get_raw_item(index=index)
        
        image = self.get_image(item['img_path'])
        
        expr = item['exprs']
        bboxes = item['bboxes']
        if self.image_transform:
            image = self.image_transform(image)

        num_boxes = len(bboxes)
        
        question = self.get_template()

        locs = ""

        for i in range(num_boxes):
            locs += f"{i}-th {expr} <trigger> "
        
        question = question.replace(PLACE_HOLDERS['expression'], expr)
        
        ret = {
            'image': image,
            'target': {
                'width': item['width'],
                'height': item['height'],
                'all_boxes': bboxes,
                'target_boxes': bboxes,
                'exprs': expr
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'{locs}',
                }
            ]
        }
        
        return ret
    

# f_p = "/data/cad-recruit-02_814/kilee/NextChat/data/coco_det_train2014.json"
# tp = "/data/cad-recruit-02_814/kilee/NextChat/dataset/template/mscoco_detection.json"
# img_dir = "/data/datasets_802/coco/images/train2014"
# test = MSCOCO(filename=f_p, template_file=tp, image_folder=img_dir)
# sample = test.__getitem__(0)
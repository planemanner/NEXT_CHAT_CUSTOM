from .base_dataset import BaseDataset
from .configs import PLACE_HOLDERS, NEW_SPECIAL_TOKENS


class VisualGenome(BaseDataset):
    def __init__(self, filename, 
                 template_file,
                 image_transform=None,
                 image_folder=None, seed=42):
        super().__init__(filename, template_file, image_folder, seed)
        # For VisualGenome and RefCOCOg
        self.image_transform = image_transform

    def __getitem__(self, index):
        item = self.get_raw_item(index=index)
        image = self.get_image(item['img_path'])
        expr = item['expression']
        bbox = [item['bbox']]
        if self.image_transform:
            image = self.image_transform(image)
        
        # Question template 중 하나를 Sampling 하고, expr token 을 dataset 내 expression 과 교체
        question = self.get_template().replace(PLACE_HOLDERS['objects'], NEW_SPECIAL_TOKENS['loc_holder'])
                
        ret = {
            'image': image, 
            'target': {
                'width': item['width'],
                'height': item['height'],
                'all_boxes': bbox,
                'target_boxes': [],
                'exprs': expr,
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'{expr}',
                    
                }
            ]
        }
        return ret


class VisualGenomeDet(BaseDataset):
    def __init__(self, filename, 
                 template_file,
                 image_transform=None,
                 image_folder=None, seed=42):
        super().__init__(filename, template_file, image_folder, seed)
        # For VisualGenome and RefCOCOg
        self.image_transform = image_transform

    def __getitem__(self, index):
        item = self.get_raw_item(index=index)
        image = self.get_image(item['img_path'])
        if self.image_transform:
            image = self.image_transform(image)

        obj_class = item['exprs']
        bboxes = item['bboxes']
        num_boxes = len(bboxes)
        
        # Question template 중 하나를 Sampling 하고, expr token 을 dataset 내 expression 과 교체
        question = self.get_template()

        locs = ""
        for i in range(num_boxes):
            locs += f"{obj_class} <trigger> "

        question = question.replace(PLACE_HOLDERS['expressions'], obj_class)

        ret = {
            'image': image, 
            'target': {
                'width': item['width'],
                'height': item['height'],
                'all_boxes': bboxes,
                'target_boxes': bboxes,
                'exprs': obj_class,
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'Answer: {locs}',
                    
                }
            ]
        }
        return ret
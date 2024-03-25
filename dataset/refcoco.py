from .base_dataset import BaseDataset
from .configs import PLACE_HOLDERS, NEW_SPECIAL_TOKENS


class RefCOCO(BaseDataset):
    def __init__(self, filename, 
                 template_file,
                 image_transform=None,
                 image_folder=None, seed=42):
        super().__init__(filename, template_file, image_folder, seed)
        self.image_transform = image_transform
        # RefCOCO, RefCOCO+

    def __getitem__(self, index):
        item = self.get_raw_item(index=index)
        image = self.get_image(item['img_path'])
        if self.image_transform:
            image = self.image_transform(image)
        expr = item['expression']
        bbox = item['bbox']

        # Question template 중 하나를 Sampling 하고, expr token 을 dataset 내 expression 과 교체
        question = self.get_template().replace(PLACE_HOLDERS['expression'], expr)

        ret = {
            'image': image,
            'target': {
                'width': item['width'],
                'height': item['height'],
                'all_boxes': [bbox],
                'target_boxes': [bbox],
                'exprs': expr
            },
            'conversations': [
                {
                    'from': 'human',
                    'value': question,
                },
                {
                    'from': 'gpt',
                    'value': f'It is at <trigger> ',
                    'boxes_seq': [[0]],
                }
            ]
        }
        
        return ret
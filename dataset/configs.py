PREDEFINED_SPECIAL_TOKENS = {
    'pad_token': '<pad>', 
    'bos_token': '<s>',
    'eos_token': '</s>',
    'unk_token': '<unk>',
}

NEW_SPECIAL_TOKENS={
    
    'img_patch': "<im_patch>", 
    'trigger': '<trigger>', # Trigger token
    'box': '<boxes>',
    'user': '<USER>',
    'system': '<ASSISTANT>',
    'do_caption': '<caption>',
    'end_prefix': '<end of prefix>',
    'do_detection': '<detect>',
    'loc_holder' : '<loc>'

}



PLACE_HOLDERS = {'image': '<image>',
                 'objects': '<objs>',
                 'bbox': '<bbox>',
                 'bboxes': '<boxes>',
                 'question': '<question>',
                 'expression': '<expr>',
                 'expressions': '<exprs>'
                 }


MAX_TOKEN_LEN = 320
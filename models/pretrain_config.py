gpt2_args = {
    'pretrain_model': 'openai-community/gpt2',
    'last_hidden_dim': 768,

}
swin_args = {
    'pretrain_model': 'microsoft/swin-base-patch4-window12-384',
    'last_hidden_dim': 1024,
    'num_patches': 144
}
# from transformers import AutoTokenizer, GPT2LMHeadModel
# import torch

# tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
# model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# print(inputs['input_ids'])
# # print(inputs.keys())
# # print(inputs['attention_mask'])
# outputs = model(**inputs)
# # print(outputs.keys())
# # logits = outputs.logits
# # print(logits.shape)

# vocab = tokenizer.get_vocab()
# for token_name, token_id in vocab.items():
    
#     if "<|endoftext|>" == token_name:
#         print(f"End-of-Text Token ID: {token_id}")
    
#     if "<bos_token>" == token_name:
#         print(f"BOS Token Exists: {token_id}")
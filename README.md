# About this repository
  - This is the toy version of NExT-Chat -> [Project Page](https://next-chatv.github.io/)
    - [Original Repository](https://github.com/NExT-ChatV/NExT-Chat)
  - The paper suggests very curious idea and give us inspiration for future research.
# To do
  # Repair :hammer:
    - Multi-Turn Conversation 형태가 가능하도록 구성
  - Conversion to Pytorch Lightening
# Repair : hammer :
  - Triggering token 구성 방식 변형

# Usage
  - Required package installation
    - ```pip install -r requirements.txt```
  - Training
    
    - Go to the scripts directory
      ```{sh}
      cd scripts
      ```
    - Authorize the shell script to be executable status
      ```{sh}
      chmod +x train.sh
      ```
    - Run the shell script
      ```{sh}
      sh train.sh
      ```
    - If you want to use more bigger multi-node ddp training, please modify train.sh as following
      ```{sh}
      #!/bin/bash

      python -m torch.distributed.launch --nproc_per_node=<the number of gpus to be used in each node> --nnodes=<the number of gpu cluster nodes> --node_rank=0 --master_addr="127.0.0.1" --master_port=25001 ../main_train.py
      ```
      
  - Evaluation
    - Go to the scripts directory
      ```{sh}
      cd scripts
      ```
    - Authorize the evaluation shell scripts to be executable status
      ```{sh}
      chmod +x cap_eval.sh
      chmod +x det_eval.sh
      ```
    - Run the shell script
      ```{sh}
      sh cap_eval.sh
      sh det_eval.sh
      ```
    - Arguments
      - cap_eval.sh
        - VAL_IMG_DIR 와 CAPTION_ANN_PATH 의 연도를 바꿔주시고, torchvision 의 CocoDetection Class 의 __getitem__ 부분을 다음과 같이 바꿔주시면 됩니다.
        ```python3
        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            id = self.ids[index]
            image = self._load_image(id)
            target = self._load_target(id)

            if self.transforms is not None:
                image, target = self.transforms(image, target)

            return image, id
        ```
      - det_eval.sh
        - VAL_FILE 를 바꿔주시면 됩니다.
          - refcoco : /data/cad-recruit-02_814/kilee/NextChat/data/REC_refcoco_unc_val.jsonl
          - refcoco+ : /data/cad-recruit-02_814/kilee/NextChat/data/REC_refcoco+_unc_val.jsonl
          - refcocog : /data/cad-recruit-02_814/kilee/NextChat/data/REC_refcoco+_umd_val.jsonl

  - Demo
    - Gradio-base demo.

    - Just run the *demo.py* ! 
      ```python3
      python demo.py
      ```

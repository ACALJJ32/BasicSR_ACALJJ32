## Installation
```
git clone https://github.com/ACALJJ32/BasicSR_ACALJJ32.git

cd BasicSR_ACALJJ32

pip install -r requirements.txt

BASICSR_EXT=True python setup.py develop
```

## Download pre-trained model
```
百度网盘
链接:https://pan.baidu.com/s/1LHYzGEqdvmrXF-Ly0Ms1lw 
提取码:awsl
```
Put this EDVR-M model in path: BasicSR_ACALJJ32/weight/edvr

## Inference
```
CUDA_VISIBLE_DEVICES=0 ./scripts/dist_demo.sh 1 ./option/demo/test_EDVR_M_x4_SR_Demo.yml [your_video_path.mp4]
```

## Change result-video scale
Just change demo config in ./options/demo/test_EDVR_M_x4_SR_Demo.yml
```
# network structures
network_g:
  ...
  scale: 4
```


## Installation
```
git clone https://github.com/ACALJJ32/BasicSR_ACALJJ32.git

cd BasicSR_ACALJJ32

pip install -r requirements.txt

BASICSR_EXT=True python setup.py develop
```

## Put your lq videos in lr_video folder

```
mkdir lr_video  # put lq vidoes
mkdir sr_video  # sr results

mkdir weight    # put pre-trained model
cd weight
mkdir edvr
```

## Download pre-trained model
```
百度网盘
链接:https://pan.baidu.com/s/1LHYzGEqdvmrXF-Ly0Ms1lw 
提取码:awsl
```
Put this EDVR-M model in path: BasicSR_ACALJJ32/weight/edvr

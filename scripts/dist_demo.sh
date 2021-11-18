#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
VIDEO=$3
PORT=${PORT:-4321}

# usage
if [ $# -ne 3 ] ;then
    echo "usage:"
    echo "./scripts/dist_test.sh [number of gpu] [path to option file]"
    exit
fi

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python basicsr/preprocess.py $VIDEO

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    basicsr/demo.py -opt $CONFIG --launcher pytorch --video_path $VIDEO

PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python basicsr/frame2video.py $VIDEO
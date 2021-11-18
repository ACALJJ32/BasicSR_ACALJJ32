import torch, os
from os import path as osp
from math import ceil
import sys
from yaml import load
from basicsr.data import build_dataloader, build_dataset
from basicsr.utils.options import parse_options
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import cv2
from copy import deepcopy
import os.path as osp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from basicsr.archs.edvr_arch import EDVR


def chop_forward(model, inp, shave=8, min_size=120000):
    # This part will divide your input in 4 small images

    b, n, c, h, w = inp.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave

    mod_size = 4
    if h_size % mod_size:
        h_size = ceil(h_size/mod_size)*mod_size  # The ceil() function returns the uploaded integer of a number
    if w_size % mod_size:
        w_size = ceil(w_size/mod_size)*mod_size

    inputlist = [
        inp[:, :, :, 0:h_size, 0:w_size],
        inp[:, :, :, 0:h_size, (w - w_size):w],
        inp[:, :, :, (h - h_size):h, 0:w_size],
        inp[:, :, :, (h - h_size):h,  (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(4):
            with torch.no_grad():
                input_batch = inputlist[i]
                output_batch = model(input_batch)
            outputlist.append(output_batch)
    else:
        outputlist = [
            chop_forward(model, patch) \
            for patch in inputlist]

    scale=4
    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output_ht = Variable(inp.data.new(b, c, h, w))

    output_ht[:, :, 0:h_half, 0:w_half] = outputlist[0][:, :, 0:h_half, 0:w_half]
    output_ht[:, :, 0:h_half, w_half:w] = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output_ht[:, :, h_half:h, 0:w_half] = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output_ht[:, :, h_half:h, w_half:w] = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output_ht

def demo_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=False)

    print("video path: ",args.video_path)

    video_name = osp.basename(args.video_path).split(".")[0]

    torch.backends.cudnn.benchmark = True

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        test_loaders.append(test_loader)

    # create model
    model_config = opt['network_g']
    _ = model_config.pop("type", "Unkown")
    model = EDVR(**model_config)

    device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
    model = model.to(device=device)

    param_key='params'
    load_net = torch.load(opt["path"].get("pretrain_network_g", "Unkown"), map_location=lambda storage, loc: storage)

    find_unused_parameters = opt.get('find_unused_parameters', False)

    model = DistributedDataParallel(
        model, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)

    # load weights
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
        load_net = load_net[param_key]

    for k, v in deepcopy(load_net).items():
        load_net['module.' + k] = v
        load_net.pop(k)

    model.load_state_dict(load_net, strict=True)
    model.eval()

    # set min size
    min_size = 921599

    # test clips
    for test_loader in test_loaders:
        for idx, data in enumerate(test_loader):
            frame_name = "{:08d}.png".format(idx)
            frame_name = osp.join("sr_video", video_name, frame_name)

            if osp.exists(frame_name): continue

            height, width = data.size()[-2:]
            if height * width < min_size:
                output = model(data)
            else:
                output = chop_forward(model, data)

            print("imwrite {:08d}.png. | totol: {}".format(idx, len(test_loader)))
            output = torch.squeeze(output.data.cpu(), dim=0).clamp(0,1).permute(1,2,0).numpy()
            cv2.imwrite(frame_name, cv2.cvtColor(output*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    demo_pipeline(root_path)

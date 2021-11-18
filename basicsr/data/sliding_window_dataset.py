import numpy as np
import random
from torch.utils import data as data
import os.path as osp
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import os
import torch
import cv2

@DATASET_REGISTRY.register()
class SlidingWindowDataset(data.Dataset):
    def __init__(self, opt):
        super(SlidingWindowDataset, self).__init__()
        self.opt = opt
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2
        self.dataroot_lq = opt["dataroot_lq"]
        self.length = len(os.listdir(self.dataroot_lq))

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        frame_name = "{:08d}".format(index)
        center_frame_idx = int(frame_name)

        # ensure not exceeding the borders
        neighbor_list = []
        seq = [x for x in range(-self.num_half_frames, self.num_half_frames + 1)]

        for index in seq:
            if index != 0:
                tmp_index = index + center_frame_idx
                if tmp_index < 0:
                    pad = 0
                elif tmp_index > self.length - 1:
                    pad = self.length - 1
                else: pad = tmp_index
                neighbor_list.append(pad)

            else: neighbor_list.append(center_frame_idx)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            img_lq_path = osp.join(self.dataroot_lq, f'{neighbor:08d}.png')
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            # numpy to tensor
            img_lq_tensor = img2tensor(img_lq)
            img_lqs.append(img_lq_tensor)

        # img_lqs: (t, c, h, w)

        return torch.stack(img_lqs, dim=0)

    def __len__(self):
        return len(os.listdir(self.dataroot_lq))
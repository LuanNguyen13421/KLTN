#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2

from utils.util import Logger
from utils.rewards import compute_reward
import utils.vsum_tools as vsum_tools
from utils.generate_dataset import Generate_Dataset
from model.summarizer_module import *

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('-i', '--input', type=str, default='', required=True, help="Input video file path")
parser.add_argument('-o', '--output', type=str, default='output', help="Output video directory")
# Misc
parser.add_argument('--seed', type=int, default=1, help="Random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="Which gpu devices to use")
parser.add_argument('--extract-method', type=str, required=True, choices=['his', 'cnn'], help="Extract feature method [Color histogram (his) / CNN (cnn)]")
parser.add_argument('--bin', type=int, default=256, help="RGB BIN (If extract feature method is color histogram)")
# Model options
parser.add_argument('--input-size', type=int, default=1024, help="Feature size expected in the input (default: 1024)")
parser.add_argument('--block-size', type=int, default=60, help="Size of blocks used inside the attention matrix (defalut: 60)")

parser.add_argument('-d', '--dataset', type=str, help="Path to h5 dataset")

parser.add_argument('--model', type=str, default='Summaries/summe/model_epoch60.pth.tar', required=True, help="Path to model file")
parser.add_argument('--save-dir', type=str, default='output/', help="Path to save output (default: 'output/')")
parser.add_argument('--use-cpu', action='store_true', help="Use cpu device")

parser.add_argument('--save-name', default='',help="'Generate video '")
parser.add_argument('--fps', type=int, default=30, help="Frames per second")
parser.add_argument('--width', type=int, default=640, help="Frame width")
parser.add_argument('--height', type=int, default=480, help="Frame height")

args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False

def main():
    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())
    test_keys = []

    for key in dataset.keys():
        test_keys.append(key)

    print("Load model")
    model = Summarizer(input_size = args.input_size, output_size = args.input_size, block_size = args.block_size)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    if args.model:
        print("Loading checkpoint from '{}'".format(args.model))
        checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    evaluate(model, dataset, test_keys, use_gpu)
    print("Summary")
    video2summary(os.path.join(args.save_dir,'result.h5'),args.input ,args.save_dir)

def evaluate(model, dataset, test_keys, use_gpu):
    with torch.no_grad():
        model.eval()

        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)

        h5_res = h5py.File(os.path.join(args.save_dir,'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq)
            if use_gpu: seq = seq.cuda()
            probs = model(seq)
            probs = probs.unsqueeze(0)
            probs = probs.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            video_name = dataset[key]['video_name'][()]
            fps = dataset[key]['fps'][()]

            sum = 0
            for i in range(len(nfps)):
                sum += nfps[i]

            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            h5_res.create_dataset(key + '/score', data=probs)
            h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
            h5_res.create_dataset(key + '/video_name', data=video_name)
            h5_res.create_dataset(key + '/fps', data=fps)

    h5_res.close()


def frm2video(video_dir, summary, vid_writer):
    print('[INFO] Video Summary')
    video_capture = cv2.VideoCapture(video_dir)
    count = 0
    for idx, val in tqdm(enumerate(summary)):
        ret, frame = video_capture.read()
        if val == 1 and ret:
            frm = cv2.resize(frame, (args.width, args.height))
            vid_writer.write(frm)
        else:
            count += 1
    print('[OUTPUT] total {} frame, ignore {} frame'.format(len(summary)-count, count))
    
def video2summary(h5_dir, video_dir, output_dir):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    h5_res = h5py.File(h5_dir, 'r')

    for idx1 in range(len(list(h5_res.keys()))):
        key = list(h5_res.keys())[idx1]
        summary = h5_res[key]['machine_summary'][...]
        video_name = h5_res[key]['video_name'][()].decode().split('/')[-1]
        fps = h5_res[key]['fps'][()]
        vid_writer = cv2.VideoWriter(
            osp.join(output_dir, args.save_name),
            cv2.VideoWriter_fourcc('m','p','4','v'),  # Sử dụng thư viện OpenCV
            fps,
            (args.width, args.height),
        )
        frm2video(video_dir, summary, vid_writer)
        vid_writer.release()
    h5_res.close()

if __name__ == '__main__':
    name_video = args.input.split('/')[-1].split('.')[0]
    args.dataset = os.path.join(args.output, name_video + '.h5')
    args.save_name = name_video + '.mp4'
    args.save_dir = os.path.join(args.save_dir, name_video)
    if not os.path.exists(args.dataset):
        gen = Generate_Dataset(args.input, args.dataset, args.extract_method, args.bin)
        gen.generate_dataset()
        gen.h5_file.close()
    main()
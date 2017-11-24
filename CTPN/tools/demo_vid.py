#!/usr/bin/env python
#
# The codes are used for implementing CTPN for scene text detection, described in:
#
# Z. Tian, W. Huang, T. He, P. He and Y. Qiao: Detecting Text in Natural Image with
# Connectionist Text Proposal Network, ECCV, 2016.
#
# Online demo is available at: textdet.com
#
# These demo codes (with our trained model) are for text-line detection (without
# side-refiement part).
#
#
# ====== Copyright by Zhi Tian, Weilin Huang, Tong He, Pan He and Yu Qiao==========

#            Email: zhi.tian@siat.ac.cn; wl.huang@siat.ac.cn
#
#   Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
#
#

from cfg import Config as cfg
from other import draw_boxes, resize_im, CaffeModel
import cv2, os, caffe, sys
from detectors import TextProposalDetector, TextDetector
import os.path as osp
from utils.timer import Timer

NET_DEF_FILE="models/deploy.prototxt"
MODEL_FILE="models/ctpn_trained_model.caffemodel"

if len(sys.argv)>1 and sys.argv[1]=="--no-gpu":
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)

vid = 0
if len(sys.argv) == 2 and sys.argv[1] != "--no-gpu":
    if os.path.exists(sys.argv[1]):
        vid = sys.argv[1]
elif len(sys.argv) == 3:
    if os.path.exists(sys.argv[2]):
        vid = sys.argv[2]
elif len(sys.argv) > 3:
    print "Wrong parameter."
    exit()

# initialize the detectors
text_proposals_detector=TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
text_detector=TextDetector(text_proposals_detector)

timer=Timer()
cam = cv2.VideoCapture(vid)

while True:
    ret, im = cam.read()
    if not ret:
        break

    timer.tic()

    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im)

    print "Number of the detected text lines: %s"%len(text_lines)
    print "Time: %f"%timer.toc()

    im_with_text_lines=draw_boxes(im, text_lines, caption="Text Detection", wait=False)
    key = cv2.waitKey(1)
    if key == 27:
        break

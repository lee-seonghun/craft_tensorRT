import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.init as init
from craft import CRAFT
from collections import OrderedDict
#from torch2trt import torch2trt
import imgproc
import cv2
#import tensorrt as trt

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


net = CRAFT()
net.load_state_dict(copyStateDict(torch.load("weights/craft_mlt_25k.pth")))
#net.load_state_dict(torch.load("weights/craft_mlt_25k.pth"))
#image = imgproc.loadImage('./fig8/Pic_2020_10_06_231541_blockId#38.jpg')
image = imgproc.loadImage('20210507031315.jpg')
#img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 526, interpolation=cv2.INTER_LINEAR, mag_ratio=1)
#img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 2140, interpolation=cv2.INTER_LINEAR, mag_ratio=1)
img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1)
#img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 960, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
ratio_h = ratio_w = 1 / target_ratio

# preprocessing
x = imgproc.normalizeMeanVariance(img_resized)
x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
x = x.cuda() 
net = net.cuda()

net = net.eval()

y, feature = net(x)

torch.onnx.export(
    net, 
    x, 
    "craft_1280.onnx", 
    export_params=True,
    opset_version=11,
    verbose=True)

#x = torch.randn(1, 3, 928, 1280).cuda()
#model_trt = torch2trt(net, [x], int8_mode=True, int8_calib_algorithm=trt.CalibrationAlgoType.MINMAX_CALIBRATION)
#model_trt = torch2trt(net, [x], int8_mode=True, int8_calib_algorithm=trt.CalibrationAlgoType.MINMAX_CALIBRATION)
#model_trt = torch2trt(net, [x], int8_mode=True, int8_calib_algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2)
#model_trt = torch2trt(net, [x], fp16_mode=True)
#model_trt = torch2trt(net, [x])

#torch.save(model_trt.state_dict(), 'craft_trt_int8_min_max_960_.pth')
#        "craft_trt_int8_min_max.onnx",
#torch.onnx.export(
#        model_trt,
#        x,
#        "craft_trt.onnx",
#        export_params=True,
#        )
#torch.save(model_trt.state_dict(), 'craft_trt_526_fp16.pth')
#torch.save(model_trt.state_dict(), 'craft_trt_2140_int8_minmax.pth')
#torch.save(model_trt.state_dict(), 'craft_trt_fp16.pth')
#net = torch.nn.DataParallel(net)

#net = torch.nn.DataParallel(net)
#cudnn.benchmark = False

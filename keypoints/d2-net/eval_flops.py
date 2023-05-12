import argparse

import numpy as np

import imageio

import torch

import cv2 

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net, APD2Net,SIFT_Detector
from lib.utils import preprocess_image
from lib.ap_utils import create_pooling_mask
from lib.pyramid import process_det_des
from lib.flops_counter import get_model_complexity_info

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--det', type=str, choices = ['SIFTDetector'], default='SIFTDetector',
    help='path to the dataset'
)
parser.add_argument(
    '--des', type=str, choices = ['D2Net', 'APD2Net'], default='D2Net',
    help='path to the dataset'
)

parser.add_argument(
    '--image_list_file', type=str, required=True,
    help='path to a file containing a list of images to process'
)

parser.add_argument(
    '--preprocessing', type=str, default='caffe',
    help='image preprocessing (caffe or torch)'
)
parser.add_argument(
    '--max_edge', type=int, default=1600,
    help='maximum image size at network input'
)
parser.add_argument(
    '--max_sum_edges', type=int, default=2800,
    help='maximum sum of image sizes at network input'
)

parser.add_argument(
    '--no-relu', dest='use_relu', action='store_false',
    help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)

parser.add_argument(
    '--output_stride', type=int, default=4,
    help='output stride of the model'
)
parser.add_argument(
    '--nr_keypoints', type=int, default=512,
    help='output stride of the model'
)
parser.add_argument(
    '--dilations', nargs="+", type=int,
    help='dilation strength of each pooling level'
)

parser.add_argument(
    '--gpu_id', type=int, default=0,
    help='output stride of the model'
)

args = parser.parse_args()

print(args)

device = 'cuda:' + str(args.gpu_id)

# Creating SIFT detector
det = SIFT_Detector(
        args.nr_keypoints,
        preprocessing=args.preprocessing
    )

# Creating Descriptor model
if args.des == 'D2Net':
    des = D2Net(
        use_relu=args.use_relu,
        OS=args.output_stride,
    )
    des = des.to(device)
elif args.des == 'APD2Net':
    des = APD2Net(
        use_relu=args.use_relu,
        OS=args.output_stride,
    )
    des = des.to(device)

# Process the file
with open(args.image_list_file, 'r') as f:
    lines = f.readlines()

flops = []
flops_sequence = []
last_seq = ''

#lines = lines[:]

for line in tqdm(lines, total=len(lines)):
    
    path = line.strip()

    image = imageio.imread(path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    new_width = int(resized_image.shape[1])
    new_height = int(resized_image.shape[0])
    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        scaling_factor = args.max_sum_edges / sum(resized_image.shape[: 2])
        new_width = int(new_width * scaling_factor)
        new_height = int(new_height * scaling_factor)

    # make devidable by 8
    if int(resized_image.shape[1]) % 8 != 0:
        new_width = int(new_width) - (int(new_width) % 8)
    if int(resized_image.shape[0]) % 8 != 0:
        new_height = int(new_height) - (int(new_height) % 8)

    dim = (new_width, new_height)
    resized_image = cv2.resize(
        resized_image,
        dim,
        interpolation = cv2.INTER_AREA 
    ).astype('float')

    
    current_seq = line.split('/')[2]
    if current_seq != last_seq:
        last_seq = current_seq
        if len(flops_sequence) > 0:
            assert len(flops_sequence) == 6
            flops.append(flops_sequence)
        flops_sequence = []
    
    input_image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )
    with torch.no_grad():

        image = torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                )

        saliency = det(image)

        if args.des == 'APD2Net':
            pooling_mask = create_pooling_mask(saliency, args.dilations, output_stride=args.output_stride)
            dense_features = des.dense_feature_extraction(image, pooling_mask)
            def input_constructor(input_res):
                    #batch = torch.ones(()).new_empty((1,input_res[0],input_res[1],input_res[2])).to(device)
                    return image, pooling_mask
        elif args.des == 'D2Net':
            dense_features = des.dense_feature_extraction(image)
            def input_constructor(input_res):
                    #batch = torch.ones(()).new_empty((1,input_res[0],input_res[1],input_res[2])).to(device)
                    return image

        nr_flops, _ = get_model_complexity_info(des.dense_feature_extraction, (image.shape[1], image.shape[2], image.shape[3]), input_constructor=input_constructor, custom_modules_hooks={}, as_strings=False,
                                           print_per_layer_stat=False, verbose=False)

        flops_sequence.append(nr_flops)

flops.append(flops_sequence) # last one must be appended too

flops = np.array(flops)
print(flops.shape)
print(np.mean(flops))
print(np.mean(np.std(flops, axis=1)))
        

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
from lib.pyramid import process_det_des

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
    '--model_file', type=str, default='models/d2_tf.pth',
    help='path to the full model'
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
    '--output_extension', type=str, default='.d2-net',
    help='extension for the output'
)
parser.add_argument(
    '--output_type', type=str, default='npz',
    help='output file type (npz or mat)'
)

parser.add_argument(
    '--multiscale', dest='multiscale', action='store_true',
    help='extract multiscale features'
)
parser.set_defaults(multiscale=False)

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
        model_file=args.model_file,
        use_relu=args.use_relu,
        OS=args.output_stride,
    )
    des = des.to(device)
elif args.des == 'APD2Net':
    des = APD2Net(
        model_file=args.model_file,
        use_relu=args.use_relu,
        OS=args.output_stride,
    )
    des = des.to(device)

# Process the file
with open(args.image_list_file, 'r') as f:
    lines = f.readlines()

for line in tqdm(lines, total=len(lines)):
    path = line.strip()

    image = imageio.imread(path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

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

    fact_i = image.shape[0] / resized_image.shape[0] # y
    fact_j = image.shape[1] / resized_image.shape[1] # x

    input_image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )
    with torch.no_grad():
        keypoints, scores, descriptors = process_det_des(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                det,des,
                args
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_j
    keypoints[:, 1] *= fact_i

    if args.output_type == 'npz':
        with open(path + args.output_extension, 'wb') as output_file:
            np.savez(
                output_file,
                keypoints=keypoints,
                scores=scores,
                descriptors=descriptors
            )
    elif args.output_type == 'mat':
        with open(path + args.output_extension, 'wb') as output_file:
            scipy.io.savemat(
                output_file,
                {
                    'keypoints': keypoints,
                    'scores': scores,
                    'descriptors': descriptors
                }
            )
    else:
        raise ValueError('Unknown output type.')

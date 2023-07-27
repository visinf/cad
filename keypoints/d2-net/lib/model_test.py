from lib.ap_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2 as cv



class DenseFeatureExtractionModule_OS8(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule_OS8, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, dilation=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1, dilation=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1, dilation=1)

        self.num_channels = 512

        self.use_relu = use_relu

    def forward(self, batch):
        output = self.conv1(batch)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool1(output)
        output = self.conv3(output)
        output = self.relu3(output)
        output = self.conv4(output)
        output = self.relu4(output)
        output = self.pool2(output)
        output = self.conv5(output)
        output = self.relu5(output)
        output = self.conv6(output)
        output = self.relu6(output)
        output = self.conv7(output)
        output = self.relu7(output)
        output = self.pool3(output)
        output = self.conv8(output)
        output = self.relu8(output)
        output = self.conv9(output)
        output = self.relu9(output)
        output = self.conv10(output)

        if self.use_relu:
            output = F.relu(output)
        return output

class DenseFeatureExtractionModule_OS4(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule_OS4, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, dilation=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.relu7 = nn.ReLU(inplace=True)
        #self.pool3 = nn.AvgPool2d(2, stride=1)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=2, dilation=2)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)

        self.num_channels = 512

        self.use_relu = use_relu

    def forward(self, batch):
        output = self.conv1(batch)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool1(output)
        output = self.conv3(output)
        output = self.relu3(output)
        output = self.conv4(output)
        output = self.relu4(output)
        output = self.pool2(output)
        output = self.conv5(output)
        output = self.relu5(output)
        output = self.conv6(output)
        output = self.relu6(output)
        output = self.conv7(output)
        output = self.relu7(output)
        #output = self.pool3(output)
        output = self.conv8(output)
        output = self.relu8(output)
        output = self.conv9(output)
        output = self.relu9(output)
        output = self.conv10(output)

        if self.use_relu:
            output = F.relu(output)
        return output

class DenseFeatureExtractionModule_OS2(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule_OS2, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, dilation=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.relu4 = nn.ReLU(inplace=True)
        #self.pool2 = nn.AvgPool2d(2, stride=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=2, dilation=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=2, dilation=2)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=2, dilation=2)
        self.relu7 = nn.ReLU(inplace=True)
        #self.pool3 = nn.AvgPool2d(2, stride=1)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=4, dilation=4)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=4, dilation=4)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=4, dilation=4)

        self.num_channels = 512

        self.use_relu = use_relu

    def forward(self, batch):
        output = self.conv1(batch)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool1(output)
        output = self.conv3(output)
        output = self.relu3(output)
        output = self.conv4(output)
        output = self.relu4(output)
        #output = self.pool2(output)
        output = self.conv5(output)
        output = self.relu5(output)
        output = self.conv6(output)
        output = self.relu6(output)
        output = self.conv7(output)
        output = self.relu7(output)
        #output = self.pool3(output)
        output = self.conv8(output)
        output = self.relu8(output)
        output = self.conv9(output)
        output = self.relu9(output)
        output = self.conv10(output)

        if self.use_relu:
            output = F.relu(output)
        return output

class DenseFeatureExtractionModule_OS1(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule_OS1, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        #self.pool1 = nn.AvgPool2d(2, stride=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=2, dilation=2)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=2, dilation=2)
        self.relu4 = nn.ReLU(inplace=True)
        #self.pool2 = nn.AvgPool2d(2, stride=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=4, dilation=4)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=4, dilation=4)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=4, dilation=4)
        self.relu7 = nn.ReLU(inplace=True)
        #self.pool3 = nn.AvgPool2d(2, stride=1)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=8, dilation=8)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(512, 512, 3, padding=8, dilation=8)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=8, dilation=8)

        self.num_channels = 512

        self.use_relu = use_relu


    def forward(self, batch):
        output = self.conv1(batch)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        #output = self.pool1(output)
        output = self.conv3(output)
        output = self.relu3(output)
        output = self.conv4(output)
        output = self.relu4(output)
        #output = self.pool2(output)
        output = self.conv5(output)
        output = self.relu5(output)
        output = self.conv6(output)
        output = self.relu6(output)
        output = self.conv7(output)
        output = self.relu7(output)
        #output = self.pool3(output)
        output = self.conv8(output)
        output = self.relu8(output)
        output = self.conv9(output)
        output = self.relu9(output)
        output = self.conv10(output)

        if self.use_relu:
            output = F.relu(output)
        return output













class DenseFeatureExtractionModule_AP_OS1(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule_AP_OS1, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = IrregularMaxPool2d()
        self.conv3 = GraphConv2d(64, 128, 3, number_pools=1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = GraphConv2d(128, 128, 3, number_pools=1, bias=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = IrregularMaxPool2d()
        self.conv5 = GraphConv2d(128, 256, 3, number_pools=2, bias=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = GraphConv2d(256, 256, 3, number_pools=2, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = GraphConv2d(256, 256, 3, number_pools=2, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.pool3 = IrregularMaxPool2d()
        self.conv8 = GraphConv2d(256, 512, 3, number_pools=3, bias=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = GraphConv2d(512, 512, 3, number_pools=3, bias=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = GraphConv2d(512, 512, 3, number_pools=3, bias=True)

        self.num_channels = 512

        self.use_relu = use_relu


    def forward(self, batch, pooling_mask):
        output = self.conv1(batch)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)

        output = img2graph(output)

        output = self.pool1(output, pooling_mask, 1)
        offsets = self.conv3.get_offsets(pooling_mask)

        output = self.conv3(output, offsets)
        output = self.relu3(output)
        output = self.conv4(output, offsets)
        output = self.relu4(output)

        output = self.pool2(output, pooling_mask, 2)
        offsets = self.conv5.get_offsets(pooling_mask)
        
        output = self.conv5(output, offsets)
        output = self.relu5(output)
        output = self.conv6(output, offsets)
        output = self.relu6(output)
        output = self.conv7(output, offsets)
        output = self.relu7(output)

        output = self.pool3(output, pooling_mask, 3)
        offsets = self.conv8.get_offsets(pooling_mask)

        output = self.conv8(output, offsets)
        output = self.relu8(output)
        output = self.conv9(output, offsets)
        output = self.relu9(output)
        output = self.conv10(output, offsets)

        if self.use_relu:
            output = F.relu(output)

        output = graph2img(output, pooling_mask, number_downsample=3)

        return output


class DenseFeatureExtractionModule_AP_OS2(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule_AP_OS2, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, dilation=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = IrregularMaxPool2d()
        self.conv5 = GraphConv2d(128, 256, 3, number_pools=1, bias=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = GraphConv2d(256, 256, 3, number_pools=1, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = GraphConv2d(256, 256, 3, number_pools=1, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.pool3 = IrregularMaxPool2d()
        self.conv8 = GraphConv2d(256, 512, 3, number_pools=2, bias=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = GraphConv2d(512, 512, 3, number_pools=2, bias=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = GraphConv2d(512, 512, 3, number_pools=2, bias=True)

        self.num_channels = 512

        self.use_relu = use_relu


    def forward(self, batch, pooling_mask):
        output = self.conv1(batch)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool1(output)
        output = self.conv3(output)
        output = self.relu3(output)
        output = self.conv4(output)
        output = self.relu4(output)

        output = img2graph(output)
        output = self.pool2(output, pooling_mask, 1)
        offsets = self.conv5.get_offsets(pooling_mask)
        
        output = self.conv5(output, offsets)
        output = self.relu5(output)
        output = self.conv6(output, offsets)
        output = self.relu6(output)
        output = self.conv7(output, offsets)
        output = self.relu7(output)

        output = self.pool3(output, pooling_mask, 2)
        offsets = self.conv8.get_offsets(pooling_mask)

        output = self.conv8(output, offsets)
        output = self.relu8(output)
        output = self.conv9(output, offsets)
        output = self.relu9(output)
        output = self.conv10(output, offsets)

        if self.use_relu:
            output = F.relu(output)

        output = graph2img(output, pooling_mask, number_downsample=2)

        return output


class DenseFeatureExtractionModule_AP_OS4(nn.Module):
    def __init__(self, use_relu=True, use_cuda=True):
        super(DenseFeatureExtractionModule_AP_OS4, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, dilation=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1, dilation=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.pool3 = IrregularMaxPool2d()
        self.conv8 = GraphConv2d(256, 512, 3, number_pools=1, bias=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = GraphConv2d(512, 512, 3, number_pools=1, bias=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = GraphConv2d(512, 512, 3, number_pools=1, bias=True)

        self.num_channels = 512

        self.use_relu = use_relu


    def forward(self, batch, pooling_mask):
        output = self.conv1(batch)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool1(output)
        output = self.conv3(output)
        output = self.relu3(output)
        output = self.conv4(output)
        output = self.relu4(output)
        output = self.pool2(output)
        output = self.conv5(output)
        output = self.relu5(output)
        output = self.conv6(output)
        output = self.relu6(output)
        output = self.conv7(output)
        output = self.relu7(output)

        output = img2graph(output)
        output = self.pool3(output, pooling_mask, 1)
        offsets = self.conv8.get_offsets(pooling_mask)

        output = self.conv8(output, offsets)
        output = self.relu8(output)
        output = self.conv9(output, offsets)
        output = self.relu9(output)
        output = self.conv10(output, offsets)

        if self.use_relu:
            output = F.relu(output)

        output = graph2img(output, pooling_mask, number_downsample=1)

        return output





class D2Net(nn.Module):
    def __init__(self, model_file=None, use_relu=True, use_cuda=True, OS=8):
        super(D2Net, self).__init__()

        if OS==1:
            self.dense_feature_extraction = DenseFeatureExtractionModule_OS1(
                use_relu=use_relu, use_cuda=use_cuda
            )
        elif OS==2:
            self.dense_feature_extraction = DenseFeatureExtractionModule_OS2(
                use_relu=use_relu, use_cuda=use_cuda
            )
        elif OS==4:
            self.dense_feature_extraction = DenseFeatureExtractionModule_OS4(
                use_relu=use_relu, use_cuda=use_cuda
            )
        elif OS==8:
            self.dense_feature_extraction = DenseFeatureExtractionModule_OS8(
                use_relu=use_relu, use_cuda=use_cuda
            )
        else:
            return NotImplementedError

        self.detection = HardDetectionModule()

        self.localization = HandcraftedLocalizationModule()

        if model_file is not None:
            if use_cuda:
                model_dict = torch.load(model_file)['model']
                model_dict = rename_model_dict(model_dict, 'D2Net')
                self.load_state_dict(model_dict)
            else:
                self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])

    def forward(self, batch):
        _, _, h, w = batch.size()
        dense_features = self.dense_feature_extraction(batch)

        detections = self.detection(dense_features)

        displacements = self.localization(dense_features)

        return {
            'dense_features': dense_features,
            'detections': detections,
            'displacements': displacements
        }

class APD2Net(nn.Module):
    def __init__(self, model_file=None, use_relu=True, use_cuda=True, OS=8):
        super(APD2Net, self).__init__()

        if OS==1:
            self.dense_feature_extraction = DenseFeatureExtractionModule_AP_OS1(
                use_relu=use_relu, use_cuda=use_cuda
            )
        elif OS==2:
            self.dense_feature_extraction = DenseFeatureExtractionModule_AP_OS2(
                use_relu=use_relu, use_cuda=use_cuda
            )
        elif OS==4:
            self.dense_feature_extraction = DenseFeatureExtractionModule_AP_OS4(
                use_relu=use_relu, use_cuda=use_cuda
            )
        elif OS==8:
            self.dense_feature_extraction = DenseFeatureExtractionModule_AP_OS8(
                use_relu=use_relu, use_cuda=use_cuda
            )
        else:
            return NotImplementedError

        self.detection = HardDetectionModule()

        self.localization = HandcraftedLocalizationModule()

        if model_file is not None:
            if use_cuda:
                model_dict = torch.load(model_file)['model']
                model_dict = rename_model_dict(model_dict, 'APD2Net', output_stride=OS)
                self.load_state_dict(model_dict)
            else:
                self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])

    def forward(self, batch):
        _, _, h, w = batch.size()
        dense_features = self.dense_feature_extraction(batch)

        detections = self.detection(dense_features)

        displacements = self.localization(dense_features)

        return {
            'dense_features': dense_features,
            'detections': detections,
            'displacements': displacements
        }


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected


class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]
        ).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        ).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(
            batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
        ).view(b, c, h, w)
        dj = F.conv2d(
            batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
        ).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        return torch.stack([step_i, step_j], dim=1)


class SIFT_Detector(nn.Module):
    def __init__(self, nr_keypoints, preprocessing='caffe'):
        super(SIFT_Detector, self).__init__()
                
        #self.sift = cv.SIFT_create(nfeatures=nr_keypoints)
        self.sift = cv.SIFT_create()
        self.nr_keypoints = nr_keypoints
        self.preprocessing = preprocessing

    def get_keypoints_from_saliency(self, saliency_map):
        assert saliency_map.shape[0] == 1
        saliency_map = saliency_map[0]
        idxs = torch.nonzero(saliency_map).t()[1:3]
        y,x = idxs[0], idxs[1]
        return y,x

    def forward(self, batch):
        B = batch.shape[0]
        saliencies = []
        for i in range(B):
            image_torch = batch[i].unsqueeze(0)
            image = batch.detach().cpu().numpy()
            if self.preprocessing == 'caffe':
                mean = np.array([103.939, 116.779, 123.68])
                image = image + mean.reshape([3, 1, 1])
                # RGB -> BGR
                image = image[:: -1, :, :]
            elif self.preprocessing == 'torch':
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image * std.reshape([3, 1, 1]) + mean.reshape([3, 1, 1])
                image *= 255.0

            image = 0.299*image[:,0,:,:] + 0.587*image[:,1,:,:] + 0.114*image[:,2,:,:]
            image = torch.from_numpy(image)
            image = image.unsqueeze(1)

            image = image.permute(0,2,3,1)[0]
            image = image.detach().cpu().numpy().astype('uint8')

            kps = self.sift.detect(image,None)
            kps = list(kps)
            y = []
            x = []
            unique_points = set()
            
            kps.sort(key=lambda x: x.response, reverse=True)
            for kp in kps:
                kp_y = round(kp.pt[1])
                kp_x = round(kp.pt[0])
                
                if (kp_y, kp_x) in unique_points:
                    continue
                else:
                    unique_points.add((kp_y, kp_x))
                
                y.append(kp_y)
                x.append(kp_x)

                if len(unique_points) == self.nr_keypoints:
                    break
                
            y = torch.tensor(y).long()
            x = torch.tensor(x).long()

            saliency = torch.zeros_like(image_torch[0,:1])
            saliency[0,y,x] = 1.
            saliencies.append(saliency)
        saliency = torch.cat(saliencies, dim = 0)
        saliency = saliency.unsqueeze(0)
        
        return saliency


def rename_model_dict(model_dict, model_name, output_stride=1):
    dense_feature_extraction_components = ['conv1', 'relu1', 'conv2', 'relu2', 'pool1', 'conv3', 'relu3', 'conv4', 'relu4',
                                                   'pool2', 'conv5', 'relu5', 'conv6', 'relu6', 'conv7', 'relu7', 'pool3', 'conv8',
                                                   'relu8', 'conv9', 'relu9', 'conv10']

    if output_stride == 1:        
        AP_ignore_list = ['conv1', 'conv2']
    if output_stride == 2:        
        AP_ignore_list = ['conv1', 'conv2', 'conv3', 'conv4']
    if output_stride == 4:        
        AP_ignore_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7']


    new_model_dict = {}
    if model_name == 'D2Net':

        for key in model_dict:

            key_components = key.split('.')
            key_components.remove(key_components[1])
            component = dense_feature_extraction_components[int(key_components[1])]
            key_components[1] = component
            new_key = ".".join(key_components)
            new_model_dict[new_key] = model_dict[key]

    elif model_name == 'APD2Net':

        for key in model_dict:

            key_components = key.split('.')
            key_components.remove(key_components[1])
            component = dense_feature_extraction_components[int(key_components[1])]
            key_components[1] = component
            if not key_components[1] in AP_ignore_list:
                key_components[1] = key_components[1] + '.conv'
                if key_components[2] == 'weight':
                    value = model_dict[key].view(model_dict[key].shape[0], model_dict[key].shape[1], 9, 1)
                else:
                    value = model_dict[key]
            else:
                value = model_dict[key]
            new_key = ".".join(key_components)
            new_model_dict[new_key] = value
    
    else:
        return NotImplemented
    
    return new_model_dict
  
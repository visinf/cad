import torch
import torchvision
import torch.nn as nn


def img2graph(img):
    return img.flatten(start_dim=2)

def get_nearest_neighbors(pooling_mask, number_downsample):
    
    B,C,H,W = pooling_mask.shape
    nearest_neighbors = torch.zeros((1,1,H,W)).long().to(pooling_mask.device) # each entry has the flat index of its nearest valid neighbor that is in the same 'pooling box'
    valid_points = torch.zeros((1,1,H,W)).long().to(pooling_mask.device)
    valid_coords = []
    start_idx = 0
    for i in range(number_downsample+1):
        
        nearest_neighbors_for_current_scale = torch.zeros_like(nearest_neighbors).float()
        # get part of graph that corresponds to the scale
        if i != number_downsample:
            pooling_mask_current_scale_highres = pooling_mask == i
        else: # If number_downsample is smaller than largest value in pooling mask pretend that all higher values belong to current scale
            pooling_mask_current_scale_highres = pooling_mask >= i
        
        number_elements = (pooling_mask_current_scale_highres).float().sum() / 4**(i)
        
        if number_elements == 0:
            continue
        
        nearest_neighbors_for_current_scale[:,:,2**(i)//2::2**(i),2**(i)//2::2**(i)] = 1. # get center pixel of each pooling box
        nearest_neighbors_for_current_scale = pooling_mask_current_scale_highres * nearest_neighbors_for_current_scale # only keep centers of current scale. 
        valid_coords.append(torch.nonzero(nearest_neighbors_for_current_scale)[:,2:4]) # remove batch and channel which are always 0 (because B=C=1)
        valid_points = valid_points + nearest_neighbors_for_current_scale
        nearest_neighbors_for_current_scale[nearest_neighbors_for_current_scale != 0] = torch.arange(start_idx, number_elements + start_idx).to(pooling_mask.device)
        if i != 0:
            mp = nn.MaxPool2d(2**(i), stride=1).to(pooling_mask.device)
            pad = torch.nn.ZeroPad2d((2**(i)//2-1, 2**(i)//2, 2**(i)//2-1, 2**(i)//2))
            nearest_neighbors_for_current_scale = pad(nearest_neighbors_for_current_scale)
            nearest_neighbors_for_current_scale = mp(nearest_neighbors_for_current_scale)
        
        nearest_neighbors = nearest_neighbors + nearest_neighbors_for_current_scale # add neighbors of current scale to all nearest neighbors
        start_idx += number_elements
    return nearest_neighbors.long(), torch.cat(valid_coords, dim=0)

class GraphConv2d(nn.Module):

    #kernel size should be uneven to guarantee symmetry
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation = 1, number_pools=0, padding_mode='zeros', bias=False):
        super(GraphConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.number_pools = number_pools
        self.padding_mode = padding_mode # 'zeros' or 'replicate'
        self.bias = bias
        
        self.conv = torchvision.ops.DeformConv2d(self.in_channels, self.out_channels, kernel_size = (kernel_size**2,1), stride = 1, padding = (kernel_size**2//2,0), dilation = 1, bias = self.bias)

        
    def forward(self, input, offsets):
        
        input = input.unsqueeze(-1) # 1,C,N,1
        output = self.conv(input, offsets) # 1,C,N,1
        output = output.squeeze(-1)
        
        return output
    
    def get_offsets(self, pooling_mask):
        nearest_neighbors, valid_indices_yx = get_nearest_neighbors(pooling_mask, self.number_pools)
        
        idxs = torch.arange(0, valid_indices_yx.shape[0]).to(pooling_mask.device)

        
        _,_,H,W = pooling_mask.shape
        #print('valid_locations', valid_locations)
        
        #valid_indices_yx = valid_locations.nonzero()[:,2:4] # remove batch and channel which are always 0 (because B=C=1)
            
        # get sampling coordinates based on dilation, coordinates of valid locations, and num_pool
        #o = 1 * self.dilation * 2**self.number_pools
        #sampling_points = torch.tensor( [[-o,   0.,  o, -o,   0.,  o, -o,   0.,  o],
        #                                 [-o, -o, -o,   0.,   0.,   0.,  o,  o,  o]]).to(valid_locations.device) # x,y
        
        max_sampling_offset = self.kernel_size // 2 * self.dilation * 2**self.number_pools
        step = max_sampling_offset*2 / (self.kernel_size - 1)

        sampling_points_single = torch.arange(-max_sampling_offset, max_sampling_offset+1e-5, step).to(pooling_mask.device)
        sampling_points_x = sampling_points_single.repeat(self.kernel_size).unsqueeze(0)
        sampling_points_y = sampling_points_single.repeat_interleave(self.kernel_size).unsqueeze(0)
        sampling_points = torch.cat([sampling_points_x, sampling_points_y], dim=0)
        
        sampling_indices_y = valid_indices_yx[:,0].unsqueeze(1) + sampling_points[1,:].unsqueeze(0)
        sampling_indices_x = valid_indices_yx[:,1].unsqueeze(1) + sampling_points[0,:].unsqueeze(0)
        
        sampling_indices_y = sampling_indices_y.round().long()
        sampling_indices_x = sampling_indices_x.round().long()
        
        #clip values outside of image, this corresponds to 'replicate', default should be 'zero' I believe
        sampling_indices_y_clip = sampling_indices_y.clip(0, H-1)
        sampling_indices_x_clip = sampling_indices_x.clip(0, W-1)
        
        # use nearest neighbor locations to get corresponding index of input (flat_lowres)
        nearest_neighbors_locations_flat_lowres = nearest_neighbors[0,:,sampling_indices_y_clip, sampling_indices_x_clip].squeeze(0) # N,self.kernel_size**2
        
        if self.padding_mode == 'zeros':
            padding_locations_y = sampling_indices_y != sampling_indices_y_clip
            padding_locations_x = sampling_indices_x != sampling_indices_x_clip
            padding_locations = torch.logical_or(padding_locations_y, padding_locations_x)
            nearest_neighbors_locations_flat_lowres[padding_locations] = -1. # aparrently 'illigal' locations are set to 0 by default in deformable conv
            
            
        
        offsets_y = nearest_neighbors_locations_flat_lowres-idxs.unsqueeze(-1)

        sampling_points_def_conv = torch.arange(-self.kernel_size**2 // 2 + 1, self.kernel_size**2 // 2+1e-5, 1.).to(pooling_mask.device)
        offsets_y = offsets_y - sampling_points_def_conv # N, self.kernel_size**2
        offsets_y = offsets_y.permute(1,0).unsqueeze(0).unsqueeze(-1) # 1, self.kernel_size**2, N, 1
        offsets_x = torch.zeros_like(offsets_y)
        offsets_y = offsets_y.unsqueeze(2)
        offsets_x = offsets_x.unsqueeze(2) # 1,self.kernel_size**2,1,N,1 # this is needed for appending x offsets=0 and then reshaping to have them in alternating order
        
        offsets = torch.cat([offsets_y, offsets_x], dim=2) # 1,self.kernel_size**2,2,N,1
        offsets = offsets.reshape(1,self.kernel_size**2 * 2,-1,1)
        
        return offsets



def graph2img(input, pooling_mask, number_downsample):
    nearest_neighbors, _= get_nearest_neighbors(pooling_mask, number_downsample)
    return input[:,:,nearest_neighbors.squeeze(0).squeeze(0)]


class IrregularMaxPool2d(nn.Module):

    def __init__(self, kernel_size=2, stride=2, padding=0, dilation = 1, return_indices: bool = False, ceil_mode: bool = False):
        super(IrregularMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.max_pool = torch.nn.MaxPool2d(kernel_size, stride=stride, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    # input: graph that corresponds to feature map of size C, H_in, W_in + some higher res pixels
    # pooling_mask: 1,1,H_in,W_in mask defining what regions should be how often downsampled/pooled. 1 means no downsampling, 2 one time, 3 two times, ...
    # must be so that entries are always in a block form that corresponds to the downpooling size
    # number_pool: INT - the number of pooling operations that the current pooling is
    # the graph has first the most high res elements than the second high res elements and so on...
    def forward(self, input, pooling_mask, number_pool):        
        

        # separate higher res elements from elements of current resolution (the ones that have been downsampled equally often)
        start_idx = 0
        for i in range(0,number_pool-1):
            start_idx += ((pooling_mask == i).float().sum() / 4**(i)).long()

        graph_higher_res = input[:,:,:start_idx]
        graph_current_res = input[:,:,start_idx:]
        
        #graph_current_res contains all element of current res. Some should be kept and others downsampled
        
        # make pooling mask to current resolution so I can work in this 'coordinate' system
        pooling_mask_current_res = pooling_mask[:,:,::2**(number_pool-1),::2**(number_pool-1)]        

        _,_,H,W = pooling_mask_current_res.shape
        _,C,_ = input.shape 
        pooling_input = torch.ones((1,C,H,W)).to(input.device) * float('-inf') # input of pooling operator, default -INF
        
        
        current_res_mask = pooling_mask_current_res >= number_pool - 1 # locations of all the valid pixels in the current scale image
        current_res_mask_flat = current_res_mask.flatten() # used to index masks to make them the same shape as current_res_graph
        
        # get elements of current scale that should not be downsampled
        dont_touch_mask = (pooling_mask_current_res == number_pool -1) # True at locations of current scale that should not be downsampled
        dont_touch_mask = dont_touch_mask[current_res_mask]
        output_dont_touch = graph_current_res[:,:,dont_touch_mask]
        
        # get elements of current scale that should be downsampled
        touch_mask = (pooling_mask_current_res >= number_pool)[:,:,::2,::2]
        _,C,H,W = pooling_input.shape
        pooling_input = pooling_input.view(1,C,-1)
        pooling_input[:,:,current_res_mask.flatten()] = graph_current_res
        pooling_input = pooling_input.view(1,C,H,W)
        pooled = self.max_pool(pooling_input)
        pooled = pooled.flatten(start_dim=-2)
        output_lower_res = pooled[:,:,touch_mask.flatten()]
        
        output_graph = torch.cat([graph_higher_res, output_dont_touch, output_lower_res], dim=2)
        
        return output_graph


def mask2poolingMask(mask, number_pools):
    B,C,H,W = mask.shape
    assert B == 1
    assert C == 1
    assert H % (2**number_pools) == 0
    assert W % (2**number_pools) == 0
    
    mask_patches = torch.nn.functional.unfold(mask, kernel_size=2**number_pools, stride=2**number_pools) # B, 4, nr   

    mask_patches_with_mask = mask_patches.sum(dim=1).squeeze(0)
    mask_patches_with_mask = (mask_patches_with_mask != 0.)
    mask_patches[:,:,mask_patches_with_mask] = 1.
    
    fold = nn.Fold(output_size=(H,W), kernel_size=(2**number_pools, 2**number_pools), stride = 2**number_pools)
    out = fold(mask_patches)
    out = (out == 0).float() # invert because 1 means it get downpooled
    out = out * number_pools
    return out

def create_pooling_mask(saliency, dils=[3,3,3], output_stride=1):
    
    if output_stride == 1:
        max_pools = 3
    if output_stride == 2:
        max_pools = 2
    if output_stride == 4:
        max_pools = 1

    device = saliency.device

    pooling_mask = torch.ones((saliency.shape)).to(device) * max_pools
    pooling_mask = pooling_mask[:,:,::output_stride,::output_stride]
    for i in reversed(range(max_pools)):
        if dils[i] != 0:

            dilation = nn.Conv2d(1, 1, dils[i], padding=dils[i]//2, bias=False).to(device)
            weight = torch.ones((1,1,dils[i],dils[i])).to(device)
            dilation.weight = nn.Parameter(weight)

            dil_mask = dilation(saliency)

            dil_mask = dil_mask[:,:,::output_stride,::output_stride]
            dil_pooling_mask = mask2poolingMask(dil_mask, i+1)
            pooling_mask[dil_pooling_mask == 0] = i
    return pooling_mask
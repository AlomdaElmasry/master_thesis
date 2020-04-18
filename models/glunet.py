from os import path as osp
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _quadruple
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import math
import re
import cupy


def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        # torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid  # here also channel first
            if not output_channel_first:
                map = map.permute(0, 2, 3, 1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float()  # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid  # here also channel first
            if not output_channel_first:
                map = map.permute(1, 2, 0).float()
        return map.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                map[i, :, :, 0] = flow[i, :, :, 0] + X
                map[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                map = map.transpose(0, 3, 1, 2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.permute(1, 2, 0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            map[:, :, 0] = flow[:, :, 0] + X
            map[:, :, 1] = flow[:, :, 1] + Y
            if output_channel_first:
                map = map.transpose(2, 0, 1).float()
        return map.astype(np.float32)


def convert_mapping_to_flow(map, output_channel_first=True):
    if not isinstance(map, np.ndarray):
        # torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            B, C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if map.is_cuda:
                grid = grid.cuda()
            flow = map - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(0, 2, 3, 1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float()  # attention, concat axis=0 here

            if map.is_cuda:
                grid = grid.cuda()

            flow = map - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(1, 2, 0).float()
        return flow.float()
    else:
        # here numpy arrays
        if len(map.shape) == 4:
            if map.shape[3] != 2:
                # size is Bx2xHxW
                map = map.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                flow[i, :, :, 0] = map[i, :, :, 0] - X
                flow[i, :, :, 1] = map[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0, 3, 1, 2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.permute(1, 2, 0)
            # HxWx2
            h_scale, w_scale = map.shape[:2]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:, :, 0] = map[:, :, 0] - X
            flow[:, :, 1] = map[:, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(2, 0, 1).float()
        return flow.astype(np.float32)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:, 0, :, :] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
    mapping[:, 1, :, :] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow


class Stream:
    ptr = torch.cuda.current_stream().cuda_stream


# end

kernel_Correlation_rearrange = '''
	extern "C" __global__ void kernel_Correlation_rearrange(
		const int n,
		const float* input,
		float* output
	) {
	  int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
	  if (intIndex >= n) {
	    return;
	  }
	  int intSample = blockIdx.z;
	  int intChannel = blockIdx.y;
	  float dblValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];
	  __syncthreads();
	  int intPaddedY = (intIndex / SIZE_3(input)) + 4;
	  int intPaddedX = (intIndex % SIZE_3(input)) + 4;
	  int intRearrange = ((SIZE_3(input) + 8) * intPaddedY) + intPaddedX;
	  output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = dblValue;
	}
'''

kernel_Correlation_updateOutput = '''
	extern "C" __global__ void kernel_Correlation_updateOutput(
	  const int n,
	  const float* rbot0,
	  const float* rbot1,
	  float* top
	) {
	  extern __shared__ char patch_data_char[];

	  float *patch_data = (float *)patch_data_char;

	  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
	  int x1 = blockIdx.x + 4;
	  int y1 = blockIdx.y + 4;
	  int item = blockIdx.z;
	  int ch_off = threadIdx.x;

	  // Load 3D patch into shared shared memory
	  for (int j = 0; j < 1; j++) { // HEIGHT
	    for (int i = 0; i < 1; i++) { // WIDTH
	      int ji_off = (j + i) * SIZE_3(rbot0);
	      for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	        int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
	        int idxPatchData = ji_off + ch;
	        patch_data[idxPatchData] = rbot0[idx1];
	      }
	    }
	  }

	  __syncthreads();

	  __shared__ float sum[32];

	  // Compute correlation
	  for (int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
	    sum[ch_off] = 0;

	    int s2o = top_channel % 9 - 4;
	    int s2p = top_channel / 9 - 4;

	    for (int j = 0; j < 1; j++) { // HEIGHT
	      for (int i = 0; i < 1; i++) { // WIDTH
	        int ji_off = (j + i) * SIZE_3(rbot0);
	        for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	          int x2 = x1 + s2o;
	          int y2 = y1 + s2p;

	          int idxPatchData = ji_off + ch;
	          int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;

	          sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
	        }
	      }
	    }

	    __syncthreads();

	    if (ch_off == 0) {
	      float total_sum = 0;
	      for (int idx = 0; idx < 32; idx++) {
	        total_sum += sum[idx];
	      }
	      const int sumelems = SIZE_3(rbot0);
	      const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
	      top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
	    }
	  }
	}
'''

kernel_Correlation_updateGradFirst = '''
	#define ROUND_OFF 50000
	extern "C" __global__ void kernel_Correlation_updateGradFirst(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradFirst); // channels
	  int l = (intIndex / SIZE_1(gradFirst)) % SIZE_3(gradFirst) + 4; // w-pos
	  int m = (intIndex / SIZE_1(gradFirst) / SIZE_3(gradFirst)) % SIZE_2(gradFirst) + 4; // h-pos

	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = round_off;

	  // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	  int xmin = (l - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)
	  int ymin = (m - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)

	  // Same here:
	  int xmax = (l - 4 + round_off_s1) - round_off; // floor (l - 4)
	  int ymax = (m - 4 + round_off_s1) - round_off; // floor (m - 4)

	  float sum = 0;
	  if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	    xmin = max(0,xmin);
	    xmax = min(SIZE_3(gradOutput)-1,xmax);

	    ymin = max(0,ymin);
	    ymax = min(SIZE_2(gradOutput)-1,ymax);

	    for (int p = -4; p <= 4; p++) {
	      for (int o = -4; o <= 4; o++) {
	        // Get rbot1 data:
	        int s2o = o;
	        int s2p = p;
	        int idxbot1 = ((intSample * SIZE_1(rbot0) + (m+s2p)) * SIZE_2(rbot0) + (l+s2o)) * SIZE_3(rbot0) + n;
	        float bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]

	        // Index offset for gradOutput in following loops:
	        int op = (p+4) * 9 + (o+4); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);

	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot1tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradFirst);
	  const int bot0index = ((n * SIZE_2(gradFirst)) + (m-4)) * SIZE_3(gradFirst) + (l-4);
	  gradFirst[bot0index + intSample*SIZE_1(gradFirst)*SIZE_2(gradFirst)*SIZE_3(gradFirst)] = sum / (float)sumelems;
	} }
'''

kernel_Correlation_updateGradSecond = '''
	#define ROUND_OFF 50000
	extern "C" __global__ void kernel_Correlation_updateGradSecond(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradSecond); // channels
	  int l = (intIndex / SIZE_1(gradSecond)) % SIZE_3(gradSecond) + 4; // w-pos
	  int m = (intIndex / SIZE_1(gradSecond) / SIZE_3(gradSecond)) % SIZE_2(gradSecond) + 4; // h-pos

	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = round_off;

	  float sum = 0;
	  for (int p = -4; p <= 4; p++) {
	    for (int o = -4; o <= 4; o++) {
	      int s2o = o;
	      int s2p = p;

	      //Get X,Y ranges and clamp
	      // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	      int xmin = (l - 4 - s2o + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4 - s2o)
	      int ymin = (m - 4 - s2p + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4 - s2o)

	      // Same here:
	      int xmax = (l - 4 - s2o + round_off_s1) - round_off; // floor (l - 4 - s2o)
	      int ymax = (m - 4 - s2p + round_off_s1) - round_off; // floor (m - 4 - s2p)

	      if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	        xmin = max(0,xmin);
	        xmax = min(SIZE_3(gradOutput)-1,xmax);

	        ymin = max(0,ymin);
	        ymax = min(SIZE_2(gradOutput)-1,ymax);

	        // Get rbot0 data:
	        int idxbot0 = ((intSample * SIZE_1(rbot0) + (m-s2p)) * SIZE_2(rbot0) + (l-s2o)) * SIZE_3(rbot0) + n;
	        float bot0tmp = rbot0[idxbot0]; // rbot1[l+s2o,m+s2p,n]

	        // Index offset for gradOutput in following loops:
	        int op = (p+4) * 9 + (o+4); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);

	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot0tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradSecond);
	  const int bot1index = ((n * SIZE_2(gradSecond)) + (m-4)) * SIZE_3(gradSecond) + (l-4);
	  gradSecond[bot1index + intSample*SIZE_1(gradSecond)*SIZE_2(gradSecond)*SIZE_3(gradSecond)] = sum / (float)sumelems;
	} }
'''


def cupy_kernel(strFunction, objectVariables):
    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel


# end

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


# end

class _FunctionCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(self, first, second):
        rbot0 = first.new_zeros([first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1)])
        rbot1 = first.new_zeros([first.size(0), first.size(2) + 8, first.size(3) + 8, first.size(1)])

        self.save_for_backward(first, second, rbot0, rbot1)

        assert (first.is_contiguous() == True)
        assert (second.is_contiguous() == True)

        output = first.new_zeros([first.size(0), 81, first.size(2), first.size(3)])

        if first.is_cuda == True:
            n = first.size(2) * first.size(3)
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
                'input': first,
                'output': rbot0
            }))(
                grid=tuple([int((n + 16 - 1) / 16), first.size(1), first.size(0)]),
                block=tuple([16, 1, 1]),
                args=[n, first.data_ptr(), rbot0.data_ptr()],
                stream=Stream
            )

            n = second.size(2) * second.size(3)
            cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
                'input': second,
                'output': rbot1
            }))(
                grid=tuple([int((n + 16 - 1) / 16), second.size(1), second.size(0)]),
                block=tuple([16, 1, 1]),
                args=[n, second.data_ptr(), rbot1.data_ptr()],
                stream=Stream
            )

            n = output.size(1) * output.size(2) * output.size(3)
            cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
                'rbot0': rbot0,
                'rbot1': rbot1,
                'top': output
            }))(
                grid=tuple([output.size(3), output.size(2), output.size(0)]),
                block=tuple([32, 1, 1]),
                shared_mem=first.size(1) * 4,
                args=[n, rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr()],
                stream=Stream
            )

        elif first.is_cuda == False:
            raise NotImplementedError()

        # end

        return output

    # end

    @staticmethod
    def backward(self, gradOutput):
        first, second, rbot0, rbot1 = self.saved_tensors

        assert (gradOutput.is_contiguous() == True)

        gradFirst = first.new_zeros([first.size(0), first.size(1), first.size(2), first.size(3)]) if \
            self.needs_input_grad[0] == True else None
        gradSecond = first.new_zeros([first.size(0), first.size(1), first.size(2), first.size(3)]) if \
            self.needs_input_grad[1] == True else None

        if first.is_cuda == True:
            if gradFirst is not None:
                for intSample in range(first.size(0)):
                    n = first.size(1) * first.size(2) * first.size(3)
                    cupy_launch('kernel_Correlation_updateGradFirst',
                                cupy_kernel('kernel_Correlation_updateGradFirst', {
                                    'rbot0': rbot0,
                                    'rbot1': rbot1,
                                    'gradOutput': gradOutput,
                                    'gradFirst': gradFirst,
                                    'gradSecond': None
                                }))(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(),
                              gradFirst.data_ptr(), None],
                        stream=Stream
                    )
            # end
            # end

            if gradSecond is not None:
                for intSample in range(first.size(0)):
                    n = first.size(1) * first.size(2) * first.size(3)
                    cupy_launch('kernel_Correlation_updateGradSecond',
                                cupy_kernel('kernel_Correlation_updateGradSecond', {
                                    'rbot0': rbot0,
                                    'rbot1': rbot1,
                                    'gradOutput': gradOutput,
                                    'gradFirst': None,
                                    'gradSecond': gradSecond
                                }))(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[n, intSample, rbot0.data_ptr(), rbot1.data_ptr(), gradOutput.data_ptr(), None,
                              gradSecond.data_ptr()],
                        stream=Stream
                    )
            # end
        # end

        elif first.is_cuda == False:
            raise NotImplementedError()

        # end

        return gradFirst, gradSecond


def FunctionCorrelation(tensorFirst, tensorSecond):
    return _FunctionCorrelation.apply(tensorFirst, tensorSecond)


class ModuleCorrelation(torch.nn.Module):
    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    def forward(self, tensorFirst, tensorSecond):
        return _FunctionCorrelation.apply(tensorFirst, tensorSecond)


class ResNetPyramid(nn.Module):
    def __init__(self, train=False):
        super().__init__()
        self.n_levels = 5
        self.model = models.resnet101(pretrained=True)
        modules = OrderedDict()
        n_block = 0

        self.resnet_module_list = [self.model.conv1,
                                   self.model.bn1,
                                   self.model.relu,
                                   self.model.maxpool,
                                   self.model.layer1,
                                   self.model.layer2,
                                   self.model.layer3,
                                   self.model.layer4]

        modules['level_0'] = nn.Sequential(*[self.model.conv1,
                                             self.model.bn1,
                                             self.model.relu])  # H_2
        for param in modules['level_0'].parameters():
            param.requires_grad = train

        modules['level_1'] = nn.Sequential(*[self.model.maxpool,
                                             self.model.layer1])  # H_4
        for param in modules['level_1'].parameters():
            param.requires_grad = train

        modules['level_2'] = nn.Sequential(*[self.model.layer2])  # H/8
        for param in modules['level_2'].parameters():
            param.requires_grad = train
        modules['level_3'] = nn.Sequential(*[self.model.layer3])  # H/16
        for param in modules['level_3'].parameters():
            param.requires_grad = train
        modules['level_4'] = nn.Sequential(*[self.model.layer4])  # H/32
        for param in modules['level_4'].parameters():
            param.requires_grad = train
        self.__dict__['_modules'] = modules

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        outputs = []
        if quarter_resolution_only:
            x_half = self.__dict__['_modules']['level_0'](x)
            x_quarter = self.__dict__['_modules']['level_1'](x_half)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_half = self.__dict__['_modules']['level_0'](x)
            x_quarter = self.__dict__['_modules']['level_1'](x_half)
            outputs.append(x_quarter)
            x_eight = self.__dict__['_modules']['level_2'](x_quarter)
            outputs.append(x_eight)
        else:
            for layer_n in range(0, self.n_levels):
                x = self.__dict__['_modules']['level_' + str(layer_n)](x)
                outputs.append(x)

            x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            outputs.append(x)
            # it will contain [H/2, H/4, H/8, H/16, H/32, H/64]
        return outputs


class VGGPyramid(nn.Module):
    def __init__(self, train=False):
        super().__init__()
        self.n_levels = 5
        source_model = models.vgg16(pretrained=True)

        modules = OrderedDict()
        tmp = []
        n_block = 0
        first_relu = False

        for c in source_model.features.children():
            if (isinstance(c, nn.ReLU) and not first_relu) or (isinstance(c, nn.MaxPool2d)):
                first_relu = True
                tmp.append(c)
                modules['level_' + str(n_block)] = nn.Sequential(*tmp)
                for param in modules['level_' + str(n_block)].parameters():
                    param.requires_grad = train

                tmp = []
                n_block += 1
            else:
                tmp.append(c)

            if n_block == self.n_levels:
                break

        self.__dict__['_modules'] = modules

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        outputs = []
        if quarter_resolution_only:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            outputs.append(x_full)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
            x_eight = self.__dict__['_modules']['level_' + str(3)](x_quarter)
            outputs.append(x_eight)
        else:
            for layer_n in range(0, self.n_levels):
                x = self.__dict__['_modules']['level_' + str(layer_n)](x)
                outputs.append(x)

            x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            outputs.append(x)
            x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='area')
            outputs.append(x)
        return outputs


class CorrelationVolume(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self):
        super(CorrelationVolume, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()

        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)  # shape (b,c,h*w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)  # shape (b,h*w,c)
        feature_mul = torch.bmm(feature_B, feature_A)  # shape (b,h*w,h*w)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor  # shape (b,h*w,h,w)


class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class OpticalFlowEstimator(nn.Module):

    def __init__(self, in_channels, batch_norm):
        super(OpticalFlowEstimator, self).__init__()

        dd = np.cumsum([128, 128, 96, 64, 32])
        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_1 = conv(in_channels + dd[0], 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_2 = conv(in_channels + dd[1], 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_3 = conv(in_channels + dd[2], 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_4 = conv(in_channels + dd[3], 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow = predict_flow(in_channels + dd[4])

    def forward(self, x):
        # dense net connection
        x = torch.cat((self.conv_0(x), x), 1)
        x = torch.cat((self.conv_1(x), x), 1)
        x = torch.cat((self.conv_2(x), x), 1)
        x = torch.cat((self.conv_3(x), x), 1)
        x = torch.cat((self.conv_4(x), x), 1)
        flow = self.predict_flow(x)
        return x, flow


class OpticalFlowEstimatorNoDenseConnection(nn.Module):

    def __init__(self, in_channels, batch_norm):
        super(OpticalFlowEstimatorNoDenseConnection, self).__init__()
        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_1 = conv(128, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_2 = conv(128, 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_3 = conv(96, 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_4 = conv(64, 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow = predict_flow(32)

    def forward(self, x):
        x = self.conv_4(self.conv_3(self.conv_2(self.conv_1(self.conv_0(x)))))
        flow = self.predict_flow(x)
        return x, flow


# extracted from DGCNet
def conv_blck(in_channels, out_channels, kernel_size=3,
              stride=1, padding=1, dilation=1, bn=False):
    if bn:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation),
                             nn.ReLU(inplace=True))


def conv_head(in_channels):
    return nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)


class CorrespondenceMapBase(nn.Module):
    def __init__(self, in_channels, bn=False):
        super().__init__()

    def forward(self, x1, x2=None, x3=None):
        x = x1
        # concatenating dimensions
        if (x2 is not None) and (x3 is None):
            x = torch.cat((x1, x2), 1)
        elif (x2 is None) and (x3 is not None):
            x = torch.cat((x1, x3), 1)
        elif (x2 is not None) and (x3 is not None):
            x = torch.cat((x1, x2, x3), 1)

        return x


class CMDTop(CorrespondenceMapBase):
    def __init__(self, in_channels, bn=False):
        super().__init__(in_channels, bn)
        chan = [128, 128, 96, 64, 32]
        self.conv0 = conv_blck(in_channels, chan[0], bn=bn)
        self.conv1 = conv_blck(chan[0], chan[1], bn=bn)
        self.conv2 = conv_blck(chan[1], chan[2], bn=bn)
        self.conv3 = conv_blck(chan[2], chan[3], bn=bn)
        self.conv4 = conv_blck(chan[3], chan[4], bn=bn)
        self.final = conv_head(chan[-1])

    def forward(self, x1, x2=None, x3=None):
        x = super().forward(x1, x2, x3)
        x = self.conv4(self.conv3(self.conv2(self.conv1(self.conv0(x)))))
        return self.final(x)


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


def Softmax1D(x, dim):
    x_k = torch.max(x, dim)[0].unsqueeze(dim)
    x -= x_k.expand_as(x)
    exp_x = torch.exp(x)
    return torch.div(exp_x, torch.sum(exp_x, dim).unsqueeze(dim).expand_as(x))


def conv4d(data, filters, bias=None, permute_filters=True, use_half=False):
    b, c, h, w, d, t = data.size()

    data = data.permute(2, 0, 1, 3, 4, 5).contiguous()  # permute to avoid making contiguous inside loop

    # Same permutation is done with filters, unless already provided with permutation
    if permute_filters:
        filters = filters.permute(2, 0, 1, 3, 4, 5).contiguous()  # permute to avoid making contiguous inside loop

    c_out = filters.size(1)
    if use_half:
        output = Variable(torch.HalfTensor(h, b, c_out, w, d, t), requires_grad=data.requires_grad)
    else:
        output = Variable(torch.zeros(h, b, c_out, w, d, t), requires_grad=data.requires_grad)

    padding = filters.size(0) // 2
    if use_half:
        Z = Variable(torch.zeros(padding, b, c, w, d, t).half())
    else:
        Z = Variable(torch.zeros(padding, b, c, w, d, t))

    if data.is_cuda:
        Z = Z.cuda(data.get_device())
        output = output.cuda(data.get_device())

    data_padded = torch.cat((Z, data, Z), 0)

    for i in range(output.size(0)):  # loop on first feature dimension
        # convolve with center channel of filter (at position=padding)
        output[i, :, :, :, :, :] = F.conv3d(data_padded[i + padding, :, :, :, :, :],
                                            filters[padding, :, :, :, :, :], bias=bias, stride=1, padding=padding)
        # convolve with upper/lower channels of filter (at postions [:padding] [padding+1:])
        for p in range(1, padding + 1):
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(data_padded[i + padding - p, :, :, :, :, :],
                                                                           filters[padding - p, :, :, :, :, :],
                                                                           bias=None, stride=1, padding=padding)
            output[i, :, :, :, :, :] = output[i, :, :, :, :, :] + F.conv3d(data_padded[i + padding + p, :, :, :, :, :],
                                                                           filters[padding + p, :, :, :, :, :],
                                                                           bias=None, stride=1, padding=padding)

    output = output.permute(1, 2, 0, 3, 4, 5).contiguous()
    return output


class Conv4d(_ConvNd):
    """Applies a 4D convolution over an input signal composed of several input
    planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, pre_permuted_filters=True):
        # stride, dilation and groups !=1 functionality not tested
        stride = 1
        dilation = 1
        groups = 1
        # zero padding is added automatically in conv4d function to preserve tensor size
        padding = 0
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)
        super(Conv4d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _quadruple(0), groups, bias)
        # weights will be sliced along one dimension during convolution loop
        # make the looping dimension to be the first one in the tensor,
        # so that we don't need to call contiguous() inside the loop
        self.pre_permuted_filters = pre_permuted_filters
        if self.pre_permuted_filters:
            self.weight.data = self.weight.data.permute(2, 0, 1, 3, 4, 5).contiguous()
        self.use_half = False

    def forward(self, input):
        return conv4d(input, self.weight, bias=self.bias, permute_filters=not self.pre_permuted_filters,
                      use_half=self.use_half)  # filters pre-permuted in constructor


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_source, feature_target):
        # feature_A is source, B is target
        if self.shape == '3D':
            # the usual correlation
            b, c, h, w = feature_source.size()
            # reshape features for matrix multiplication
            feature_source = feature_source.transpose(2, 3).contiguous().view(b, c, h * w)
            feature_target = feature_target.view(b, c, h * w).transpose(1, 2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_target, feature_source)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
            correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        elif self.shape == '4D':
            b, c, hsource, wsource = feature_source.size()
            b, c, htarget, wtarget = feature_target.size()
            # reshape features for matrix multiplication
            feature_source = feature_source.view(b, c, hsource * wsource).transpose(1, 2)  # size [b,hsource*wsource,c]
            feature_target = feature_target.view(b, c, htarget * wtarget)  # size [b,c,htarget*wtarget]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_source, feature_target)  # size [b, hsource*wsource, htarget*wtarget]
            correlation_tensor = feature_mul.view(b, hsource, wsource, htarget, wtarget).unsqueeze(1)
            # size is [b, 1, hsource, wsource, htarget, wtarget]

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor


class NeighConsensus(torch.nn.Module):
    def __init__(self, use_cuda=True, kernel_sizes=[3, 3, 3], channels=[10, 10, 1], symmetric_mode=True):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i == 0:
                ch_in = 1
            else:
                ch_in = channels[i - 1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(Conv4d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, bias=True))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 4, 5, 2, 3)
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x = self.conv(x)
        return x


def MutualMatching(corr4d):
    # mutual matching
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B] #correlation target
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (corr4d_A * corr4d_B)  # parenthesis are important for symmetric output

    return corr4d


def maxpool4d(corr4d_hres, k_size=4):
    slices = []
    for i in range(k_size):
        for j in range(k_size):
            for k in range(k_size):
                for l in range(k_size):
                    slices.append(corr4d_hres[:, 0, i::k_size, j::k_size, k::k_size, l::k_size].unsqueeze(0))
    slices = torch.cat(tuple(slices), dim=1)
    corr4d, max_idx = torch.max(slices, dim=1, keepdim=True)
    max_l = torch.fmod(max_idx, k_size)
    max_k = torch.fmod(max_idx.sub(max_l).div(k_size), k_size)
    max_j = torch.fmod(max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size), k_size)
    max_i = max_idx.sub(max_l).div(k_size).sub(max_k).div(k_size).sub(max_j).div(k_size)
    # i,j,k,l represent the *relative* coords of the max point in the box of size k_size*k_size*k_size*k_size
    return (corr4d, max_i, max_j, max_k, max_l)


class GLUNet_model(nn.Module):
    '''
    GLU-Net
    '''

    def __init__(self, evaluation, div=1.0, iterative_refinement=False,
                 refinement_at_all_levels=False, refinement_at_adaptive_reso=True,
                 batch_norm=True, pyramid_type='VGG', md=4, upfeat_channels=2, dense_connection=True,
                 consensus_network=False, cyclic_consistency=True, decoder_inputs='corr_flow_feat'):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(GLUNet_model, self).__init__()
        self.div = div
        self.pyramid_type = pyramid_type
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.l2norm = FeatureL2Norm()
        self.iterative_refinement = iterative_refinement  # only during evaluation

        # where to put the refinement networks
        self.refinement_at_all_levels = refinement_at_all_levels
        self.refinement_at_adaptive_reso = refinement_at_adaptive_reso

        # definition of the inputs to the decoders
        self.decoder_inputs = decoder_inputs
        self.dense_connection = dense_connection
        self.upfeat_channels = upfeat_channels

        # improvement of the global correlation
        self.cyclic_consistency = cyclic_consistency
        self.consensus_network = consensus_network
        if self.cyclic_consistency:
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
        elif consensus_network:
            ncons_kernel_sizes = [3, 3, 3]
            ncons_channels = [10, 10, 1]
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
            # normalisation is applied in code here
            self.NeighConsensus = NeighConsensus(use_cuda=True,
                                                 kernel_sizes=ncons_kernel_sizes,
                                                 channels=ncons_channels)
        else:
            self.corr = CorrelationVolume()

        dd = np.cumsum([128, 128, 96, 64, 32])
        # 16x16
        nd = 16 * 16  # global correlation
        od = nd + 2
        self.decoder4 = CMDTop(in_channels=od, bn=batch_norm)
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # 32x32
        nd = (2 * md + 1) ** 2  # constrained correlation, 4 pixels on each side
        if self.decoder_inputs == 'corr_flow_feat':
            od = nd + 2
        elif self.decoder_inputs == 'corr':
            od = nd
        elif self.decoder_inputs == 'corr_flow':
            od = nd + 2
        if dense_connection:
            self.decoder3 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = od + dd[4]
        else:
            self.decoder3 = OpticalFlowEstimatorNoDenseConnection(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = 32

        # weights for refinement module
        if self.refinement_at_all_levels or self.refinement_at_adaptive_reso:
            self.dc_conv1 = conv(input_to_refinement, 128, kernel_size=3, stride=1, padding=1, dilation=1,
                                 batch_norm=batch_norm)
            self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
            self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
            self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
            self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
            self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
            self.dc_conv7 = predict_flow(32)

        # 1/8 of original resolution
        nd = (2 * md + 1) ** 2  # constrained correlation, 4 pixels on each side
        if self.decoder_inputs == 'corr_flow_feat':
            od = nd + 2
        elif self.decoder_inputs == 'corr':
            od = nd
        elif self.decoder_inputs == 'corr_flow':
            od = nd + 2
        if dense_connection:
            self.decoder2 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = od + dd[4]
        else:
            self.decoder2 = OpticalFlowEstimatorNoDenseConnection(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = 32
        if self.decoder_inputs == 'corr_flow_feat':
            self.upfeat2 = deconv(input_to_refinement, self.upfeat_channels, kernel_size=4, stride=2, padding=1)

        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        if refinement_at_all_levels:
            # weights for refinement module
            self.dc_conv1_level2 = conv(input_to_refinement, 128, kernel_size=3, stride=1, padding=1, dilation=1,
                                        batch_norm=batch_norm)
            self.dc_conv2_level2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
            self.dc_conv3_level2 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
            self.dc_conv4_level2 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
            self.dc_conv5_level2 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
            self.dc_conv6_level2 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
            self.dc_conv7_level2 = predict_flow(32)

        # 1/4 of original resolution
        nd = (2 * md + 1) ** 2  # constrained correlation, 4 pixels on each side
        if self.decoder_inputs == 'corr_flow_feat':
            od = nd + self.upfeat_channels + 2
        elif self.decoder_inputs == 'corr':
            od = nd
        elif self.decoder_inputs == 'corr_flow':
            od = nd + 2
        if dense_connection:
            self.decoder1 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = od + dd[4]
        else:
            self.decoder1 = OpticalFlowEstimatorNoDenseConnection(in_channels=od, batch_norm=batch_norm)
            input_to_refinement = 32

        self.l_dc_conv1 = conv(input_to_refinement, 128, kernel_size=3, stride=1, padding=1, dilation=1,
                               batch_norm=batch_norm)
        self.l_dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.l_dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.l_dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.l_dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.l_dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.l_dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        if pyramid_type == 'ResNet':
            self.pyramid = ResNetPyramid()
        else:
            self.pyramid = VGGPyramid()

        self.evaluation = evaluation

    def pre_process_data(self, source_img, target_img, device, apply_flip=False):
        '''
        :param source_img:
        :param target_img:
        :param apply_flip:
        :param device:
        :return:
        '''

        # img has shape bx3xhxw
        b, _, h_original, w_original = target_img.shape
        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])

        # original resolution
        if h_original < 256:
            int_preprocessed_height = 256
        else:
            int_preprocessed_height = int(math.floor(int(h_original / 8.0) * 8.0))

        if w_original < 256:
            int_preprocessed_width = 256
        else:
            int_preprocessed_width = int(math.floor(int(w_original / 8.0) * 8.0))

        if apply_flip:
            # if apply flip, horizontally flip the target images
            target_img_original = target_img
            target_img = []
            for i in range(b):
                transformed_image = np.fliplr(target_img_original[i].cpu().permute(1, 2, 0).numpy())
                target_img.append(transformed_image)

            target_img = torch.from_numpy(np.uint8(target_img)).permute(0, 3, 1, 2)

        source_img_copy = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area').byte()
        target_img_copy = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area').byte()
        source_img_copy = source_img_copy.float().div(255.0)
        target_img_copy = target_img_copy.float().div(255.0)
        mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])

        # resolution 256x256
        source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                         size=(256, 256),
                                                         mode='area').byte()
        target_img_256 = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                         size=(256, 256),
                                                         mode='area').byte()

        source_img_256 = source_img_256.float().div(255.0)
        target_img_256 = target_img_256.float().div(255.0)
        source_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])

        ratio_x = float(w_original) / float(int_preprocessed_width)
        ratio_y = float(h_original) / float(int_preprocessed_height)

        return source_img_copy.to(device), target_img_copy.to(device), source_img_256.to(device), target_img_256.to(
            device), \
               ratio_x, ratio_y, h_original, w_original

    def flipping_condition(self, im_source_base, im_target_base, device):

        # should only happen during evaluation
        target_image_is_flipped = False  # for training
        if not self.evaluation:
            raise ValueError('Flipping condition should only happen during evaluation')
        else:
            list_average_flow = []
            false_true = [False, True]
            for apply_flipping in false_true:
                im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, h_base, w_base = \
                    self.pre_process_data(im_source_base, im_target_base, apply_flip=apply_flipping, device=device)
                b, _, h_256, w_256 = im_target_256.size()

                with torch.no_grad():
                    # pyramid, 256 reso
                    im1_pyr_256 = self.pyramid(im_target_256)
                    im2_pyr_256 = self.pyramid(im_source_256)
                    c14 = im1_pyr_256[-3]
                    c24 = im2_pyr_256[-3]

                flow4 = self.coarsest_resolution_flow(c14, c24, h_256, w_256)
                average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
                list_average_flow.append(average_flow.item())
            target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
            if target_image_is_flipped:
                list_average_flow = []
                # if previous way found that target is flipped with respect to the source ==> check that the
                # other way finds the same thing
                # ==> the source becomes the target and the target becomes source
                for apply_flipping in false_true:
                    im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, h_base, w_base = \
                        self.pre_process_data(im_target_base, im_source_base, apply_flip=apply_flipping, device=device)
                    b, _, h_256, w_256 = im_target_256.size()

                    with torch.no_grad():
                        # pyramid, 256 reso
                        im1_pyr_256 = self.pyramid(im_target_256)
                        im2_pyr_256 = self.pyramid(im_source_256)
                        c14 = im1_pyr_256[-3]
                        c24 = im2_pyr_256[-3]

                    flow4 = self.coarsest_resolution_flow(c14, c24, h_256, w_256)
                    average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
                    list_average_flow.append(average_flow.item())
                target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
                # if the right direction found that it is flipped, either the other direction finds the same,
                # then it is flipped, otherwise it isnt flipped

        # found out if better to flip the target image or not, now pre-process the new source and target images
        self.target_image_is_flipped = target_image_is_flipped
        im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, \
        h_original, w_original = self.pre_process_data(im_source_base, im_target_base,
                                                       apply_flip=target_image_is_flipped, device=device)
        return im_source.to(device), im_target.to(device), im_source_256.to(device), im_target_256.to(device), \
               ratio_x, ratio_y, h_original, w_original

    def coarsest_resolution_flow(self, c14, c24, h_256, w_256):
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)
        b = c24.shape[0]
        if self.cyclic_consistency:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4 = corr4d.squeeze(1).view(b, c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        elif self.consensus_network:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4d = self.NeighConsensus(corr4d)
            corr4d = MutualMatching(corr4d)  # size is [b, 1, hsource, wsource, htarget, wtarget]
            corr4 = corr4d.squeeze(1).view(c24.shape[0], c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        else:
            corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))
        corr4 = self.l2norm(F.relu(corr4))
        b, c, h, w = corr4.size()
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        est_map4 = self.decoder4(x1=corr4, x3=init_map)
        # conversion to flow and from there constrained correlation
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4) / self.div
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        return flow4

    def forward(self, im_target, im_source, im_target_256, im_source_256):
        # all indices 1 refer to target images
        # all indices 2 refer to source images

        b, _, h_full, w_full = im_target.size()
        b, _, h_256, w_256 = im_target_256.size()
        div = self.div

        # extract pyramid features
        with torch.no_grad():
            im1_pyr = self.pyramid(im_target, eigth_resolution=True)
            im2_pyr = self.pyramid(im_source, eigth_resolution=True)
            c11 = im1_pyr[-2]  # size original_res/4xoriginal_res/4
            c21 = im2_pyr[-2]
            c12 = im1_pyr[-1]  # size original_res/8xoriginal_res/8
            c22 = im2_pyr[-1]

            # pyramid, 256 reso
            im1_pyr_256 = self.pyramid(im_target_256)
            im2_pyr_256 = self.pyramid(im_source_256)
            c13 = im1_pyr_256[-4]
            c23 = im2_pyr_256[-4]
            c14 = im1_pyr_256[-3]
            c24 = im2_pyr_256[-3]

        # RESOLUTION 256x256
        # level 16x16
        flow4 = self.coarsest_resolution_flow(c14, c24, h_256, w_256)
        up_flow4 = self.deconv4(flow4)

        # level 32x32
        ratio_x = 32.0 / float(w_256)
        ratio_y = 32.0 / float(h_256)
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        warp3 = warp(c23, up_flow_4_warping)
        # constrained correlation now
        corr3 = FunctionCorrelation(tensorFirst=c13, tensorSecond=warp3)
        corr3 = self.leakyRELU(corr3)
        if self.decoder_inputs == 'corr_flow_feat':
            corr3 = torch.cat((corr3, up_flow4), 1)
        elif self.decoder_inputs == 'corr':
            corr3 = corr3
        elif self.decoder_inputs == 'corr_flow':
            corr3 = torch.cat((corr3, up_flow4), 1)
        x3, res_flow3 = self.decoder3(corr3)
        flow3 = res_flow3 + up_flow4
        # flow 3 refined (at 32x32 resolution)
        if self.refinement_at_adaptive_reso or self.refinement_at_all_levels:
            x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x3))))
            flow3 = flow3 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.iterative_refinement and self.evaluation:
            # from 32x32 resolution, if upsampling to 1/8*original resolution is too big,
            # do iterative upsampling so that gap is always smaller than 2.
            R_w = float(w_full) / 8.0 / 32.0
            R_h = float(h_full) / 8.0 / 32.0
            if R_w > R_h:
                R = R_w
            else:
                R = R_h

            minimum_ratio = 3.0
            nbr_extra_layers = max(0, int(round(np.log(R / minimum_ratio) / np.log(2))))

            if nbr_extra_layers == 0:
                flow3[:, 0, :, :] *= float(w_full) / float(256)
                flow3[:, 1, :, :] *= float(h_full) / float(256)
                # ==> put the upflow in the range [Horiginal x Woriginal]
            else:
                # adding extra layers
                flow3[:, 0, :, :] *= float(w_full) / float(256)
                flow3[:, 1, :, :] *= float(h_full) / float(256)
                for n in range(nbr_extra_layers):
                    ratio = 1.0 / (8.0 * 2 ** (nbr_extra_layers - n))
                    up_flow3 = F.interpolate(input=flow3, size=(int(h_full * ratio), int(w_full * ratio)),
                                             mode='bilinear',
                                             align_corners=False)
                    c23_bis = torch.nn.functional.interpolate(c22, size=(int(h_full * ratio), int(w_full * ratio)),
                                                              mode='area')
                    c13_bis = torch.nn.functional.interpolate(c12, size=(int(h_full * ratio), int(w_full * ratio)),
                                                              mode='area')
                    warp3 = warp(c23_bis, up_flow3 * div * ratio)
                    corr3 = FunctionCorrelation(tensorFirst=c13_bis, tensorSecond=warp3)
                    corr3 = self.leakyRELU(corr3)
                    if self.decoder_inputs == 'corr_flow_feat':
                        corr3 = torch.cat((corr3, up_flow3), 1)
                    elif self.decoder_inputs == 'corr':
                        corr3 = corr3
                    elif self.decoder_inputs == 'corr_flow':
                        corr3 = torch.cat((corr3, up_flow3), 1)
                    x, res_flow3 = self.decoder2(corr3)
                    flow3 = res_flow3 + up_flow3

            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_full / 8.0), int(w_full / 8.0)), mode='bilinear',
                                     align_corners=False)
        else:
            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_full / 8.0), int(w_full / 8.0)), mode='bilinear',
                                     align_corners=False)
            up_flow3[:, 0, :, :] *= float(w_full) / float(256)
            up_flow3[:, 1, :, :] *= float(h_full) / float(256)
            # ==> put the upflow in the range [Horiginal x Woriginal]

        # level 1/8 of original resolution
        ratio = 1.0 / 8.0
        warp2 = warp(c22, up_flow3 * div * ratio)
        corr2 = FunctionCorrelation(tensorFirst=c12, tensorSecond=warp2)
        corr2 = self.leakyRELU(corr2)
        if self.decoder_inputs == 'corr_flow_feat':
            corr2 = torch.cat((corr2, up_flow3), 1)
        elif self.decoder_inputs == 'corr':
            corr2 = corr2
        elif self.decoder_inputs == 'corr_flow':
            corr2 = torch.cat((corr2, up_flow3), 1)
        x2, res_flow2 = self.decoder2(corr2)
        flow2 = res_flow2 + up_flow3
        if self.refinement_at_all_levels:
            x = self.dc_conv4_level2(self.dc_conv3_level2(self.dc_conv2_level2(self.dc_conv1_level2(x2))))
            flow2 = flow2 + self.dc_conv7_level2(self.dc_conv6_level2(self.dc_conv5_level2(x)))

        up_flow2 = self.deconv2(flow2)
        if self.decoder_inputs == 'corr_flow_feat':
            up_feat2 = self.upfeat2(x2)

        # level 1/4 of original resolution
        ratio = 1.0 / 4.0
        warp1 = warp(c21, up_flow2 * div * ratio)
        corr1 = FunctionCorrelation(tensorFirst=c11, tensorSecond=warp1)
        corr1 = self.leakyRELU(corr1)
        if self.decoder_inputs == 'corr_flow_feat':
            corr1 = torch.cat((corr1, up_flow2, up_feat2), 1)
        elif self.decoder_inputs == 'corr':
            corr1 = corr1
        if self.decoder_inputs == 'corr_flow':
            corr1 = torch.cat((corr1, up_flow2), 1)
        x, res_flow1 = self.decoder1(corr1)
        flow1 = res_flow1 + up_flow2
        x = self.l_dc_conv4(self.l_dc_conv3(self.l_dc_conv2(self.l_dc_conv1(x))))
        flow1 = flow1 + self.l_dc_conv7(self.l_dc_conv6(self.l_dc_conv5(x)))

        if self.evaluation:
            return flow1
        else:
            return [flow4, flow3], [flow2, flow1]


class GLU_Net:
    def __init__(self, model_type='DPED_CityScape_ADE', path_pre_trained_models='pre_trained_models/',
                 apply_flipping_condition=False, pyramid_type='VGG', iterative_refinement=True,
                 feature_concatenation=False, decoder_inputs='corr_flow_feat', up_feat_channels=2,
                 cyclic_consistency=True, consensus_network=False, dense_connections=True):

        self.apply_flipping_condition = apply_flipping_condition
        # semantic glu-net
        if feature_concatenation:
            net = SemanticGLUNet_model(batch_norm=True, pyramid_type=pyramid_type,
                                       div=1.0, evaluation=True, consensus_network=consensus_network,
                                       iterative_refinement=iterative_refinement)

            if consensus_network:
                checkpoint_fname = osp.join(path_pre_trained_models, 'Semantic_GLUNet_' + model_type + '.pth')
            else:
                raise ValueError('there are no saved weights for this configuration')

        else:
            net = GLUNet_model(batch_norm=True,
                               pyramid_type=pyramid_type,
                               div=1.0, evaluation=True,
                               refinement_at_adaptive_reso=True,
                               decoder_inputs=decoder_inputs,
                               upfeat_channels=up_feat_channels,
                               dense_connection=dense_connections,
                               cyclic_consistency=cyclic_consistency,
                               consensus_network=consensus_network,
                               iterative_refinement=iterative_refinement)

            if cyclic_consistency and dense_connections and decoder_inputs == 'corr_flow_feat' and up_feat_channels == 2:
                checkpoint_fname = osp.join(path_pre_trained_models, 'GLUNet_' + model_type + '.pth')
            else:
                raise ValueError('there are no saved weights for this configuration')

        if not osp.isfile(checkpoint_fname):
            raise ValueError('check the snapshots path, checkpoint is {}'.format(checkpoint_fname))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            net.load_state_dict(torch.load(checkpoint_fname))
        except:
            net.load_state_dict(torch.load(checkpoint_fname)['state_dict'])

        print("loaded the weights")
        net.eval()
        self.net = net.to(device)  # load on GPU

    def estimate_flow(self, source_img, target_img, device, mode='channel_first'):
        if self.apply_flipping_condition:
            source_img_copy, target_img_copy, source_img_256, target_img_256, ratio_x, ratio_y, h_original, w_original \
                = self.net.flipping_condition(source_img, target_img, device)
            estimated_flow = self.net(target_img_copy, source_img_copy, target_img_256, source_img_256)
            flow_original_reso = torch.nn.functional.interpolate(input=estimated_flow, size=(h_original, w_original),
                                                                 mode='bilinear', align_corners=False)
            flow_original_reso[:, 0, :, :] *= ratio_x
            flow_original_reso[:, 1, :, :] *= ratio_y

            if self.net.target_image_is_flipped:
                flipped_mapping = convert_flow_to_mapping(flow_original_reso, output_channel_first=True) \
                    .permute(0, 2, 3, 1).cpu().numpy()
                b = flipped_mapping.shape[0]
                mapping_per_batch = []
                for i in range(b):
                    map = np.copy(np.fliplr(flipped_mapping[i]))
                    mapping_per_batch.append(map)

                mapping = torch.from_numpy(np.float32(mapping_per_batch)).permute(0, 3, 1, 2).to(device)
                flow_original_reso = convert_mapping_to_flow(mapping, device)

        else:
            source_img_copy, target_img_copy, source_img_256, target_img_256, ratio_x, ratio_y, h_original, w_original \
                = self.net.pre_process_data(source_img, target_img, device)
            estimated_flow = self.net(target_img_copy, source_img_copy,
                                      target_img_256, source_img_256)

            flow_original_reso = torch.nn.functional.interpolate(input=estimated_flow, size=(h_original, w_original),
                                                                 mode='bilinear', align_corners=False)
            flow_original_reso[:, 0, :, :] *= ratio_x
            flow_original_reso[:, 1, :, :] *= ratio_y

        if mode == 'channel_first':
            return flow_original_reso
        else:
            return flow_original_reso.permute(0, 2, 3, 1)


class GLOCAL_Net:
    def __init__(self, model_type='default', path_pre_trained_models='pre_trained_models/',
                 constrained_corr=True, global_corr=True, residual=True,
                 decoder_inputs='flow_and_feat', refinement_32=False):

        self.fixed_input = True
        if global_corr:
            if constrained_corr:
                net = GLOCALNet_model(evaluation=True, residual=residual, input_decoder=decoder_inputs,
                                      div=1, refinement=True, batch_norm=True, refinement_32=refinement_32)
                checkpoint_fname = osp.join(path_pre_trained_models, 'GLOCALNet_' + model_type + '.pth')

            else:
                net = GLOBALNet_model(evaluation=True, div=1, refinement=True, batch_norm=True)
                checkpoint_fname = osp.join(path_pre_trained_models, 'GLOBALNet_' + model_type + '.pth')
        else:
            self.fixed_input = False
            net = LOCALNet_model(evaluation=True, div=1, refinement=True, batch_norm=True)
            checkpoint_fname = osp.join(path_pre_trained_models, 'LOCALNet_' + model_type + '.pth')

        # check the checkpoints
        if not osp.isfile(checkpoint_fname):
            raise ValueError('check the snapshots path')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            net.load_state_dict(torch.load(checkpoint_fname))
        except:
            net.load_state_dict(torch.load(checkpoint_fname)['state_dict'])

        print("loaded the weights")
        net.eval()
        self.net = net.to(device)  # load on GPU

    def pre_process_data(self, source_img, target_img, device):
        # img has shape bx3xhxw
        b, _, h_scale, w_scale = target_img.shape
        if self.fixed_input:
            int_preprocessed_height = 256
            int_preprocessed_width = 256
        else:
            int_preprocessed_height = int(math.floor(math.ceil(h_scale / 16.0) * 16.0))
            int_preprocessed_width = int(math.floor(math.ceil(w_scale / 16.0) * 16.0))

        source_img_copy = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area')
        target_img_copy = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area')
        source_img_copy = source_img_copy.float().div(255.0)
        target_img_copy = target_img_copy.float().div(255.0)
        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])
        mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
        '''
        # to get exactly same results as in paper, but not on gpu
        source_img_copy = torch.zeros((b, 3, int_preprocessed_height, int_preprocessed_width))
        target_img_copy = torch.zeros((b, 3, int_preprocessed_height, int_preprocessed_width))
        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])
        normTransform = transforms.Normalize(mean_vector, std_vector)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((int_preprocessed_height, int_preprocessed_width), interpolation=2),
                                        transforms.ToTensor(),
                                        normTransform])
        for i in range(source_img.shape[0]):
            source_img_copy[i] = transform(source_img[i])
            target_img_copy[i] = transform(target_img[i])
        '''

        ratio_x = float(w_scale) / float(int_preprocessed_width)
        ratio_y = float(h_scale) / float(int_preprocessed_height)
        return source_img_copy.to(device), target_img_copy.to(device), ratio_x, ratio_y

    def estimate_flow(self, source_img, target_img, device, mode='channel_first'):
        w_scale = target_img.shape[3]
        h_scale = target_img.shape[2]
        source_img_copy, target_img_copy, ratio_x, ratio_y = self.pre_process_data(source_img.clone().detach(),
                                                                                   target_img.clone().detach(),
                                                                                   device)

        estimated_flow = torch.nn.functional.interpolate(input=self.net(target_img_copy, source_img_copy),
                                                         size=(h_scale, w_scale),
                                                         mode='bilinear', align_corners=False)

        estimated_flow[:, 0, :, :] *= ratio_x
        estimated_flow[:, 1, :, :] *= ratio_y
        # shape is Bx2xHxW here
        if mode == 'channel_first':
            return estimated_flow
        else:
            return estimated_flow.permute(0, 2, 3, 1)


class SemanticGLUNet_model(nn.Module):
    """
    Semantic-GLU-Net
    """

    def __init__(self, evaluation, div=1.0, batch_norm=True, pyramid_type='VGG', md=4,
                 cyclic_consistency=False, consensus_network=True, iterative_refinement=False):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(SemanticGLUNet_model, self).__init__()
        self.div = div
        self.pyramid_type = pyramid_type
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.iterative_refinement = iterative_refinement
        self.cyclic_consistency = cyclic_consistency
        self.consensus_network = consensus_network
        if self.cyclic_consistency:
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
        elif consensus_network:
            ncons_kernel_sizes = [3, 3, 3]
            ncons_channels = [10, 10, 1]
            self.corr = FeatureCorrelation(shape='4D', normalization=False)
            # normalisation is applied in code here

            self.NeighConsensus = NeighConsensus(use_cuda=True,
                                                 kernel_sizes=ncons_kernel_sizes,
                                                 channels=ncons_channels)
        else:
            self.corr = CorrelationVolume()
        # L2 feature normalisation
        self.l2norm = FeatureL2Norm()

        dd = np.cumsum([128, 128, 96, 64, 32])
        # weights for decoder at different levels
        nd = 16 * 16  # global correlation
        od = nd + 2
        self.decoder4 = CMDTop(in_channels=od, bn=batch_norm)
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        nd = (2 * md + 1) ** 2  # constrained correlation, 4 pixels on each side
        od = nd + 2
        self.decoder3 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)

        # weights for refinement module
        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.dc_conv7 = predict_flow(32)

        # 1/8 of original resolution
        nd = (2 * md + 1) ** 2  # constrained correlation, 4 pixels on each side
        od = nd + 2  # only gets the upsampled flow
        self.decoder2 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat2 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        # 1/4 of original resolution
        nd = (2 * md + 1) ** 2  # constrained correlation, 4 pixels on each side
        od = nd + 4
        self.decoder1 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)

        self.l_dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.l_dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.l_dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.l_dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.l_dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.l_dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.l_dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        if pyramid_type == 'ResNet':
            self.pyramid = ResNetPyramid()
        else:
            self.pyramid = VGGPyramid()

        self.evaluation = evaluation

    def pre_process_data(self, source_img, target_img, device, apply_flip=False):
        '''
        :param source_img:
        :param target_img:
        :param apply_flip:
        :param device:
        :return:
        '''

        # img has shape bx3xhxw
        b, _, h_original, w_original = target_img.shape
        mean_vector = np.array([0.485, 0.456, 0.406])
        std_vector = np.array([0.229, 0.224, 0.225])

        # original resolution
        # if image is smaller than our base network, calculate on this and then downsample to the original size
        if h_original < 256:
            int_preprocessed_height = 256
        else:
            int_preprocessed_height = int(math.floor(int(h_original / 8.0) * 8.0))

        if w_original < 256:
            int_preprocessed_width = 256
        else:
            int_preprocessed_width = int(math.floor(int(w_original / 8.0) * 8.0))

        if apply_flip:
            # flip the target image horizontally
            target_img_original = target_img
            target_img = []
            for i in range(b):
                transformed_image = np.fliplr(target_img_original[i].cpu().permute(1, 2, 0).numpy())
                target_img.append(transformed_image)

            target_img = torch.from_numpy(np.uint8(target_img)).permute(0, 3, 1, 2)

        source_img_copy = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area').byte()
        target_img_copy = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                          size=(int_preprocessed_height, int_preprocessed_width),
                                                          mode='area').byte()
        source_img_copy = source_img_copy.float().div(255.0)
        target_img_copy = target_img_copy.float().div(255.0)
        mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
        source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])

        # resolution 256x256
        source_img_256 = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                         size=(256, 256),
                                                         mode='area').byte()
        target_img_256 = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                         size=(256, 256),
                                                         mode='area').byte()

        source_img_256 = source_img_256.float().div(255.0)
        target_img_256 = target_img_256.float().div(255.0)
        source_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])
        target_img_256.sub_(mean[:, None, None]).div_(std[:, None, None])

        ratio_x = float(w_original) / float(int_preprocessed_width)
        ratio_y = float(h_original) / float(int_preprocessed_height)

        return source_img_copy.to(device), target_img_copy.to(device), source_img_256.to(device), target_img_256.to(
            device), \
               ratio_x, ratio_y, h_original, w_original

    def flipping_condition(self, im_source_base, im_target_base, device):
        # should only happen during evaluation
        target_image_is_flipped = False  # for training
        if not self.evaluation:
            raise ValueError('Flipping condition should only happen during evaluation')
        else:
            list_average_flow = []
            false_true = [False, True]
            for apply_flipping in false_true:
                im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, h_base, w_base = \
                    self.pre_process_data(im_source_base, im_target_base, apply_flip=apply_flipping, device=device)
                b, _, h_256, w_256 = im_target_256.size()

                with torch.no_grad():
                    # pyramid, 256 reso
                    im1_pyr_256 = self.pyramid(im_target_256)
                    im2_pyr_256 = self.pyramid(im_source_256)
                    c14 = im1_pyr_256[-3]
                    c24 = im2_pyr_256[-3]
                    c15 = im1_pyr_256[-2]
                    c25 = im2_pyr_256[-2]
                    c24_concat = torch.cat(
                        (c24, F.interpolate(input=c25, size=(16, 16), mode='bilinear', align_corners=False)), 1)
                    c14_concat = torch.cat(
                        (c14, F.interpolate(input=c15, size=(16, 16), mode='bilinear', align_corners=False)), 1)

                flow4 = self.coarsest_resolution_flow(c14_concat, c24_concat, h_256, w_256)
                average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
                list_average_flow.append(average_flow.item())
            target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
            if target_image_is_flipped:
                list_average_flow = []
                # if previous way found that target is flipped with respect to the source ==> check that the
                # other way finds the same thing
                # ==> the source becomes the target and the target becomes source
                for apply_flipping in false_true:
                    im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, h_base, w_base = \
                        self.pre_process_data(im_target_base, im_source_base, apply_flip=apply_flipping, device=device)
                    b, _, h_256, w_256 = im_target_256.size()

                    with torch.no_grad():
                        # pyramid, 256 reso
                        im1_pyr_256 = self.pyramid(im_target_256)
                        im2_pyr_256 = self.pyramid(im_source_256)
                        c14 = im1_pyr_256[-3]
                        c24 = im2_pyr_256[-3]
                        c15 = im1_pyr_256[-2]
                        c25 = im2_pyr_256[-2]
                        c24_concat = torch.cat(
                            (c24, F.interpolate(input=c25, size=(16, 16), mode='bilinear', align_corners=False)), 1)
                        c14_concat = torch.cat(
                            (c14, F.interpolate(input=c15, size=(16, 16), mode='bilinear', align_corners=False)), 1)

                    flow4 = self.coarsest_resolution_flow(c14_concat, c24_concat, h_256, w_256)
                    average_flow = torch.mean(torch.abs(flow4[:, 0, :, :]), dim=(1, 2))
                    list_average_flow.append(average_flow.item())
                target_image_is_flipped = false_true[np.argmin(np.float32(list_average_flow))]
                # if the right direction found that it is flipped, either the other direction finds the same,
                # then it is flipped, otherwise it isnt flipped

        self.target_image_is_flipped = target_image_is_flipped
        im_source, im_target, im_source_256, im_target_256, ratio_x, ratio_y, \
        h_original, w_original = self.pre_process_data(im_source_base, im_target_base,
                                                       apply_flip=target_image_is_flipped, device=device)
        return im_source.to(device), im_target.to(device), im_source_256.to(device), im_target_256.to(device), \
               ratio_x, ratio_y, h_original, w_original

    def coarsest_resolution_flow(self, c14, c24, h_256, w_256):
        ratio_x = 16.0 / float(w_256)
        ratio_y = 16.0 / float(h_256)
        b = c14.shape[0]
        if self.cyclic_consistency:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4 = corr4d.squeeze(1).view(b, c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        elif self.consensus_network:
            corr4d = self.corr(self.l2norm(c24), self.l2norm(c14))  # first source, then target
            # run match processing model
            corr4d = MutualMatching(corr4d)
            corr4d = self.NeighConsensus(corr4d)
            corr4d = MutualMatching(corr4d)  # size is [b, 1, hsource, wsource, htarget, wtarget]
            corr4 = corr4d.squeeze(1).view(c24.shape[0], c24.shape[2] * c24.shape[3], c14.shape[2], c14.shape[3])
        else:
            corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))
        corr4 = self.l2norm(F.relu(corr4))
        b, c, h, w = corr4.size()
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        est_map4 = self.decoder4(x1=corr4, x3=init_map)
        # conversion to flow and from there constrained correlation
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4) / self.div
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        return flow4

    def forward(self, im_target, im_source, im_target_256, im_source_256):
        # all indices 1 refer to target images
        # all indices 2 refer to source images

        b, _, h_original, w_original = im_target.size()
        b, _, h_256, w_256 = im_target_256.size()
        div = self.div

        # pyramid, original reso
        with torch.no_grad():
            im1_pyr = self.pyramid(im_target, eigth_resolution=True)
            im2_pyr = self.pyramid(im_source, eigth_resolution=True)
            c11 = im1_pyr[-2]  # size original_res/4xoriginal_res/4
            c21 = im2_pyr[-2]
            c12 = im1_pyr[-1]  # size original_res/8xoriginal_res/8
            c22 = im2_pyr[-1]

            # pyramid, 256 reso
            im1_pyr_256 = self.pyramid(im_target_256)
            im2_pyr_256 = self.pyramid(im_source_256)
            c13 = im1_pyr_256[-4]
            c23 = im2_pyr_256[-4]
            c14 = im1_pyr_256[-3]
            c24 = im2_pyr_256[-3]
            c15 = im1_pyr_256[-2]
            c25 = im2_pyr_256[-2]

        # RESOLUTION 256x256
        # level 16x16
        c24_concat = torch.cat((c24, F.interpolate(input=c25, size=(16, 16), mode='bilinear', align_corners=False)), 1)
        c14_concat = torch.cat((c14, F.interpolate(input=c15, size=(16, 16), mode='bilinear', align_corners=False)), 1)
        flow4 = self.coarsest_resolution_flow(c14_concat, c24_concat, h_256, w_256)
        up_flow4 = self.deconv4(flow4)

        # level 32x32
        ratio_x = 32.0 / float(w_256)
        ratio_y = 32.0 / float(h_256)
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        c23_concat = torch.cat((c23, F.interpolate(input=c24, size=(32, 32), mode='bilinear', align_corners=False),
                                F.interpolate(input=c25, size=(32, 32), mode='bilinear', align_corners=False)), 1)
        c13_concat = torch.cat((c13, F.interpolate(input=c14, size=(32, 32), mode='bilinear', align_corners=False),
                                F.interpolate(input=c15, size=(32, 32), mode='bilinear', align_corners=False)), 1)
        warp3 = warp(c23_concat, up_flow_4_warping)
        # constrained correlation now
        corr3 = FunctionCorrelation(tensorFirst=c13_concat, tensorSecond=warp3)
        corr3 = self.leakyRELU(corr3)
        corr3 = torch.cat((corr3, up_flow4), 1)
        x, res_flow3 = self.decoder3(corr3)
        flow3 = res_flow3 + up_flow4
        # flow 3 refined (at 32x32 resolution)
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow3 = flow3 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.evaluation and self.iterative_refinement:
            # from 32x32 resolution, if upsampling to 1/8*original resolution is too big,
            # do iterative upsampling so that gap is always smaller than 2.
            R_w = float(w_original) / 8.0 / 32.0
            R_h = float(h_original) / 8.0 / 32.0
            if R_w > R_h:
                R = R_w
            else:
                R = R_h

            minimum_ratio = 3.0
            nbr_extra_layers = max(0, int(round(np.log(R / minimum_ratio) / np.log(2))))

            if nbr_extra_layers == 0:
                flow3[:, 0, :, :] *= float(w_original) / float(256)
                flow3[:, 1, :, :] *= float(h_original) / float(256)
                # ==> put the upflow in the range [Horiginal x Woriginal]
            else:
                # adding extra layers
                flow3[:, 0, :, :] *= float(w_original) / float(256)
                flow3[:, 1, :, :] *= float(h_original) / float(256)
                for n in range(nbr_extra_layers):
                    ratio = 1.0 / (8.0 * 2 ** (nbr_extra_layers - n))
                    up_flow3 = F.interpolate(input=flow3, size=(int(h_original * ratio), int(w_original * ratio)),
                                             mode='bilinear',
                                             align_corners=False)
                    c23_bis = torch.nn.functional.interpolate(c22,
                                                              size=(int(h_original * ratio), int(w_original * ratio)),
                                                              mode='area')
                    c13_bis = torch.nn.functional.interpolate(c12,
                                                              size=(int(h_original * ratio), int(w_original * ratio)),
                                                              mode='area')
                    warp3 = warp(c23_bis, up_flow3 * div * ratio)
                    corr3 = FunctionCorrelation(tensorFirst=c13_bis, tensorSecond=warp3)
                    corr3 = self.leakyRELU(corr3)
                    corr3 = torch.cat((corr3, up_flow3), 1)
                    x, res_flow3 = self.decoder2(corr3)
                    flow3 = res_flow3 + up_flow3

            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_original / 8.0), int(w_original / 8.0)), mode='bilinear',
                                     align_corners=False)
        else:
            # ORIGINAL RESOLUTION
            up_flow3 = F.interpolate(input=flow3, size=(int(h_original / 8.0), int(w_original / 8.0)), mode='bilinear',
                                     align_corners=False)
            up_flow3[:, 0, :, :] *= float(w_original) / float(256)
            up_flow3[:, 1, :, :] *= float(h_original) / float(256)
            # ==> put the upflow in the range [Horiginal x Woriginal]

        # ORIGINAL RESOLUTION
        # level 1/8 of original resolution
        ratio = 1.0 / 8.0
        warp2 = warp(c22, up_flow3 * div * ratio)
        corr2 = FunctionCorrelation(tensorFirst=c12, tensorSecond=warp2)
        corr2 = self.leakyRELU(corr2)
        corr2 = torch.cat((corr2, up_flow3), 1)
        x, res_flow2 = self.decoder2(corr2)
        flow2 = res_flow2 + up_flow3
        up_flow2 = self.deconv2(flow2)
        up_feat2 = self.upfeat2(x)

        # level 1/4 of original resolution
        ratio = 1.0 / 4.0
        warp1 = warp(c21, up_flow2 * div * ratio)
        corr1 = FunctionCorrelation(tensorFirst=c11, tensorSecond=warp1)
        corr1 = self.leakyRELU(corr1)
        corr1 = torch.cat((corr1, up_flow2, up_feat2), 1)
        x, res_flow1 = self.decoder1(corr1)
        flow1 = res_flow1 + up_flow2
        x = self.l_dc_conv4(self.l_dc_conv3(self.l_dc_conv2(self.l_dc_conv1(x))))
        flow1 = flow1 + self.l_dc_conv7(self.l_dc_conv6(self.l_dc_conv5(x)))

        if self.evaluation:
            return flow1
        else:
            return [flow4, flow3], [flow2, flow1]


class GLOCALNet_model(nn.Module):
    '''
    GLOCAL-Net
    '''

    def __init__(self, evaluation, div=1.0, refinement=True, refinement_32=False, batch_norm=True, residual=True,
                 pyramid_type='VGG', md=4, input_decoder='flow_and_feat'):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(GLOCALNet_model, self).__init__()
        self.div = div
        self.refinement = refinement
        self.refinement_32 = refinement_32
        self.residual = residual
        self.pyramid_type = pyramid_type
        if pyramid_type == 'VGG':
            nbr_features = [512, 512, 512, 256, 128, 64, 64]
        else:
            # these are PWC-Net feature back-bone
            nbr_features = [196, 128, 96, 64, 32, 16, 3]

        self.input_decoder = input_decoder
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.corr = CorrelationVolume()

        # L2 feature normalisation
        self.l2norm = FeatureL2Norm()

        dd = np.cumsum([128, 128, 96, 64, 32])
        # weights for decoder at different levels

        nd = 16 * 16  # global correlation
        od = nd + 2
        self.decoder4 = CMDTop(in_channels=od, bn=batch_norm)
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        nd = (2 * md + 1) ** 2  # constrained correlation, 4 pixels on each side
        if self.input_decoder == 'flow_and_feat':
            od = nd + 2
        elif self.input_decoder == 'flow_and_feat_and_feature':
            od = nd + 2 + nbr_features[-4]
        elif self.input_decoder == 'feature':
            od = nd + nbr_features[-4]
        elif self.input_decoder == 'corr_only':
            od = nd
        elif self.input_decoder == 'flow':
            od = nd + 2
        self.decoder3 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        if self.refinement_32:
            self.dc_conv1_level3 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1,
                                        batch_norm=batch_norm)
            self.dc_conv2_level3 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2,
                                        batch_norm=batch_norm)
            self.dc_conv3_level3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4,
                                        batch_norm=batch_norm)
            self.dc_conv4_level3 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8,
                                        batch_norm=batch_norm)
            self.dc_conv5_level3 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16,
                                        batch_norm=batch_norm)
            self.dc_conv6_level3 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1,
                                        batch_norm=batch_norm)
            self.dc_conv7_level3 = predict_flow(32)

        nd = (2 * md + 1) ** 2  # constrained correlation, 4 pixels on each side
        if self.input_decoder == 'flow_and_feat':
            od = nd + 4
        elif self.input_decoder == 'flow_and_feat_and_feature':
            od = nd + 4 + nbr_features[-3]
        elif self.input_decoder == 'feature':
            od = nd + nbr_features[-3]
        elif self.input_decoder == 'corr_only':
            od = nd
        elif self.input_decoder == 'flow':
            od = nd + 2
        self.decoder2 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # weights for refinement module
        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        if pyramid_type == 'VGG':
            self.pyramid = VGGPyramid()
        else:
            raise ValueError("No other back-bone implemented, please choose VGG")

        self.evaluation = evaluation

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        return output * mask

    def forward(self, im_target, im_source, w_original=256, h_original=256):
        # im_target is target image ==> corresponds to all the indices 1,
        # im_source is source image ==> corresponds to all the indices 2

        b, _, h_original, w_original = im_target.size()
        div = self.div
        if self.pyramid_type == 'VGG':
            im1_pyr = self.pyramid(im_target)
            im2_pyr = self.pyramid(im_source)
            c14 = im1_pyr[-3]
            c24 = im2_pyr[-3]
            c13 = im1_pyr[-4]
            c23 = im2_pyr[-4]
            c12 = im1_pyr[-5]
            c22 = im2_pyr[-5]
        else:
            raise ValueError("No other back-bone implemented, please choose VGG")

        # level 16x16
        corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))
        corr4 = self.l2norm(F.relu(corr4))
        b, c, h, w = corr4.size()
        ratio_x = w / float(w_original)
        ratio_y = h / float(h_original)
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        est_map4 = self.decoder4(x1=corr4, x3=init_map)

        # conversion to flow and from there PWCNet
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4) / self.div
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        up_flow4 = self.deconv4(flow4)

        # level 32x32
        ratio_x = up_flow4.shape[3] / float(w_original)
        ratio_y = up_flow4.shape[2] / float(h_original)
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        warp3 = self.warp(c23, up_flow_4_warping)
        # constrained correlation now
        corr3 = FunctionCorrelation(tensorFirst=c13, tensorSecond=warp3)
        corr3 = self.leakyRELU(corr3)
        if self.input_decoder == 'flow_and_feat':
            corr3 = torch.cat((corr3, up_flow4), 1)
        elif self.input_decoder == 'flow_and_feat_and_feature':
            corr3 = torch.cat((corr3, c13, up_flow4), 1)
        elif self.input_decoder == 'feature':
            corr3 = torch.cat((corr3, c13), 1)
        elif self.input_decoder == 'corr_only':
            corr3 = corr3
        elif self.input_decoder == 'flow':
            corr3 = torch.cat((corr3, up_flow4), 1)
        x3, res_flow3 = self.decoder3(corr3)
        if self.residual:
            flow3 = res_flow3 + up_flow4
        else:
            flow3 = res_flow3
        if self.refinement_32:
            x = self.dc_conv4_level3(self.dc_conv3_level3(self.dc_conv2_level3(self.dc_conv1_level3(x3))))
            flow3 = flow3 + self.dc_conv7_level3(self.dc_conv6_level3(self.dc_conv5_level3(x)))

        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x3)

        # level 64x64
        ratio_x = up_flow3.shape[3] / float(w_original)
        ratio_y = up_flow3.shape[2] / float(h_original)
        up_flow_3_warping = up_flow3 * div
        up_flow_3_warping[:, 0, :, :] *= ratio_x
        up_flow_3_warping[:, 1, :, :] *= ratio_y
        warp2 = self.warp(c22, up_flow_3_warping)
        corr2 = FunctionCorrelation(tensorFirst=c12, tensorSecond=warp2)
        corr2 = self.leakyRELU(corr2)
        if self.input_decoder == 'flow_and_feat':
            corr2 = torch.cat((corr2, up_flow3, up_feat3), 1)
        elif self.input_decoder == 'feature':
            corr2 = torch.cat((corr2, c12), 1)
        elif self.input_decoder == 'flow_and_feat_and_feature':
            corr2 = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        elif self.input_decoder == 'corr_only':
            corr2 = corr2
        elif self.input_decoder == 'flow':
            corr2 = torch.cat((corr2, up_flow3), 1)
        x, res_flow2 = self.decoder2(corr2)
        if self.residual:
            flow2 = res_flow2 + up_flow3
        else:
            flow2 = res_flow2

        if self.refinement:
            x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
            flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.evaluation:
            return flow2
        else:
            return [flow4, flow3, flow2]


class GLOBALNet_model(nn.Module):
    '''
    GLOBAL-Net model
    '''

    def __init__(self, evaluation, div=1.0, refinement=True, batch_norm=True, pyramid_type='VGG'):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warping
        """
        super(GLOBALNet_model, self).__init__()

        self.div = div
        self.refinement = refinement
        self.pyramid_type = pyramid_type
        if pyramid_type == 'VGG':
            nbr_features = [512, 512, 512, 256, 128, 64, 64]
        else:
            # this is PWC-Net backbone
            nbr_features = [196, 128, 96, 64, 32, 16, 3]

        self.leakyRELU = nn.LeakyReLU(0.1)
        self.corr = CorrelationVolume()
        # L2 feature normalisation
        self.l2norm = FeatureL2Norm()

        dd = np.cumsum([128, 128, 96, 64, 32])
        # weights for decoder at different levels

        nd = 16 * 16  # global correlation
        od = nd + 2
        self.decoder4 = CMDTop(in_channels=od, bn=batch_norm)
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        nd = nbr_features[-4] * 2  # concatenating the features
        od = nd + 2
        self.decoder3 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        nd = nbr_features[-3] * 2  # concatenating the features
        od = nd + 4
        self.decoder2 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # weights for refinement module
        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        if pyramid_type == 'VGG':
            self.pyramid = VGGPyramid()
        else:
            raise ValueError("No other back-bone implemented, please choose VGG")

        self.evaluation = evaluation

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def forward(self, im_target, im_source, w_original=256, h_original=256):
        # im_target is target image ==> corresponds to all the indices 1,
        # im_source is source image ==> corresponds to all the indices 2

        b, _, h_original, w_original = im_target.size()
        div = self.div
        if self.pyramid_type == 'VGG':
            im1_pyr = self.pyramid(im_target)
            im2_pyr = self.pyramid(im_source)
            c14 = im1_pyr[-3]
            c24 = im2_pyr[-3]
            c13 = im1_pyr[-4]
            c23 = im2_pyr[-4]
            c12 = im1_pyr[-5]
            c22 = im2_pyr[-5]
        else:
            raise ValueError("No other back-bone implemented, please choose VGG")

        # 16x16
        corr4 = self.corr(self.l2norm(c24), self.l2norm(c14))
        corr4 = self.l2norm(F.relu(corr4))
        b, c, h, w = corr4.size()
        ratio_x = w / float(w_original)
        ratio_y = h / float(h_original)
        if torch.cuda.is_available():
            init_map = torch.FloatTensor(b, 2, h, w).zero_().cuda()
        else:
            init_map = torch.FloatTensor(b, 2, h, w).zero_()
        est_map4 = self.decoder4(x1=corr4, x3=init_map)

        # conversion to flow
        flow4 = unnormalise_and_convert_mapping_to_flow(est_map4) / self.div
        flow4[:, 0, :, :] /= ratio_x
        flow4[:, 1, :, :] /= ratio_y
        up_flow4 = self.deconv4(flow4)

        # 32x32
        ratio_x = up_flow4.shape[3] / float(w_original)
        ratio_y = up_flow4.shape[2] / float(h_original)
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        warp3 = self.warp(c23, up_flow_4_warping)
        # concatenate features now
        concat3 = torch.cat((c13, warp3, up_flow4), 1)
        x3, res_flow3 = self.decoder3(concat3)
        flow3 = res_flow3 + up_flow4
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x3)

        # 64x64
        ratio_x = up_flow3.shape[3] / float(w_original)
        ratio_y = up_flow3.shape[2] / float(h_original)
        up_flow_3_warping = up_flow3 * div
        up_flow_3_warping[:, 0, :, :] *= ratio_x
        up_flow_3_warping[:, 1, :, :] *= ratio_y
        warp2 = self.warp(c22, up_flow_3_warping)
        concat2 = torch.cat((c12, warp2, up_flow3, up_feat3), 1)
        x, res_flow2 = self.decoder2(concat2)
        flow2 = res_flow2 + up_flow3

        if self.refinement:
            x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
            flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.evaluation:
            return flow2
        else:
            return [flow4, flow3, flow2]


class LOCALNet_model(nn.Module):
    '''
    LOCAL-Net
    '''

    def __init__(self, evaluation, div=1.0, refinement=True, batch_norm=True, pyramid_type='VGG', md=4):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(LOCALNet_model, self).__init__()
        self.pyramid_type = pyramid_type
        if pyramid_type == 'VGG':
            nbr_features = [512, 512, 512, 256, 128, 64, 64]
        else:
            nbr_features = [196, 128, 96, 64, 32, 16, 3]

        self.div = div
        self.refinement = refinement
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.corr = CorrelationVolume()
        # L2 feature normalisation
        self.l2norm = FeatureL2Norm()

        dd = np.cumsum([128, 128, 96, 64, 32])
        nd = (2 * md + 1) ** 2  # constrained corr, 4 pixels on each side
        od = nd
        self.decoder4 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        nd = (2 * md + 1) ** 2  # constrained correlation, 4 pixels on each side
        od = nd + 4
        self.decoder3 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        nd = (2 * md + 1) ** 2  # constrained correlation, 4 pixels on each side
        od = nd + 4
        self.decoder2 = OpticalFlowEstimator(in_channels=od, batch_norm=batch_norm)

        # weights for refinement module
        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, batch_norm=batch_norm)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, batch_norm=batch_norm)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8, batch_norm=batch_norm)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16, batch_norm=batch_norm)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=batch_norm)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        if pyramid_type == 'VGG':
            self.pyramid = VGGPyramid()
        else:
            raise ValueError("No other back-bone implemented, please choose VGG")

        self.evaluation = evaluation

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        return output * mask

    def forward(self, im_target, im_source, w_original=256, h_original=256):
        # im_target is target image ==> corresponds to all the indices 1,
        # im_source is source image ==> corresponds to all the indices 2

        b, _, h_original, w_original = im_target.size()
        div = self.div
        if self.pyramid_type == 'VGG':
            im1_pyr = self.pyramid(im_target)
            im2_pyr = self.pyramid(im_source)
            c14 = im1_pyr[-3]
            c24 = im2_pyr[-3]
            c13 = im1_pyr[-4]
            c23 = im2_pyr[-4]
            c12 = im1_pyr[-5]
            c22 = im2_pyr[-5]
        else:
            raise ValueError("No other back-bone implemented, please choose VGG")

        # level H/16 x W/16
        corr4 = FunctionCorrelation(tensorFirst=c14, tensorSecond=c24)
        corr4 = self.leakyRELU(corr4)
        x4, flow4 = self.decoder4(corr4)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x4)

        # level H/8 x W/8
        ratio_x = up_flow4.shape[3] / float(w_original)
        ratio_y = up_flow4.shape[2] / float(h_original)
        up_flow_4_warping = up_flow4 * div
        up_flow_4_warping[:, 0, :, :] *= ratio_x
        up_flow_4_warping[:, 1, :, :] *= ratio_y
        warp3 = self.warp(c23, up_flow_4_warping)
        corr3 = FunctionCorrelation(tensorFirst=c13, tensorSecond=warp3)
        corr3 = self.leakyRELU(corr3)
        corr3 = torch.cat((corr3, up_flow4, up_feat4), 1)
        x3, res_flow3 = self.decoder3(corr3)
        flow3 = res_flow3 + up_flow4
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x3)

        # level H/2 x W/2
        ratio_x = up_flow3.shape[3] / float(w_original)
        ratio_y = up_flow3.shape[2] / float(h_original)
        up_flow_3_warping = up_flow3 * div
        up_flow_3_warping[:, 0, :, :] *= ratio_x
        up_flow_3_warping[:, 1, :, :] *= ratio_y
        warp2 = self.warp(c22, up_flow_3_warping)
        corr2 = FunctionCorrelation(tensorFirst=c12, tensorSecond=warp2)
        corr2 = self.leakyRELU(corr2)
        corr2 = torch.cat((corr2, up_flow3, up_feat3), 1)
        x, res_flow2 = self.decoder2(corr2)
        flow2 = res_flow2 + up_flow3

        if self.refinement:
            x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
            flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.evaluation:
            return flow2
        else:
            return [flow4, flow3, flow2]

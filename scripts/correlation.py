import torch


class ModuleCorrelation(torch.nn.Module):

    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    def forward(self, x_1, x_2):
        return FunctionCorrelation.apply(x_1, x_2)


class FunctionCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        # Decompose input data
        x_1, x_2 = args
        assert (x_1.is_contiguous())
        assert (x_2.is_contiguous())

        # Create zeros of h+8, w+8 (to know why) -> (B,H+8,W+8,C)
        rbot0 = x_1.new_zeros([x_1.size(0), x_1.size(2) + 8, x_1.size(3) + 8, x_1.size(1)])
        rbot1 = x_1.new_zeros([x_1.size(0), x_1.size(2) + 8, x_1.size(3) + 8, x_1.size(1)])
        output = x_1.new_zeros([x_1.shape[0], 81, x_1.shape[2], x_1.shape[3]])

        # kernel_Correlation_rearrange - input: x_1 - output: rbot0
        n = x_1.shape[2] * x_1.shape[3]
        grid_1 = tuple([int((n + 16 - 1) / 16), x_1.shape[1], x_1.shape[0]])
        block_1 = tuple([16, 1, 1])
        args_1 = [n, x_1.data_ptr(), rbot0.data_ptr()]

        a = 1
        # return _FunctionCorrelation.apply(tensorFirst, tensorSecond)

    @staticmethod
    def backward(ctx, *grad_outputs):
        pass

    @staticmethod
    def kernel_correlation_rearrange(n, input, output):
        pass


x_1 = torch.rand((1, 3, 100, 100))
x_2 = torch.rand((1, 3, 100, 100))
c = ModuleCorrelation()(x_1, x_2)

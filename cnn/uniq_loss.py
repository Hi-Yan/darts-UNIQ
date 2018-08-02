from torch.nn import CrossEntropyLoss, Module
from cnn.resnet_model_search import ResNet
from numpy import linspace, arctanh
import matplotlib.pyplot as plt
from math import tanh


class UniqLoss(Module):
    def __init__(self, lmdba, MaxBopsBits, kernel_sizes, folderName):
        super(UniqLoss, self).__init__()
        self.lmdba = lmdba
        self.search_loss = CrossEntropyLoss().cuda()
        # self.resnet_input_size = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 1]

        # build model for uniform distribution of bits
        uniform_model = ResNet(self.search_loss, bitwidths=[MaxBopsBits], kernel_sizes=kernel_sizes)
        self.maxBops = uniform_model.countBops()

        # init bops loss function
        self.bopsLoss = self.__tanh(xDst=1, yDst=0.5, yMin=0, yMax=5)
        self.plotFunction(self.bopsLoss, folderName)

    def forward(self, input, target, modelBops):
        # big penalization if bops over MaxBops
        penalization_factor = 1
        if (modelBops > self.maxBops):
            penalization_factor = 5  # TODO - change penalization factor
        quant_loss = penalization_factor * (modelBops / self.maxBops)

        return self.search_loss(input, target) + (self.lmdba * quant_loss)

    def __tanh(self, xDst, yDst, yMin, yMax):
        factor = 10

        yDelta = yMin - (-1)
        scale = yMax / (1 + yDelta)

        v = (yDst / scale) - yDelta
        v = arctanh(v)
        v /= factor
        v = round(v, 5)
        v = xDst - v

        def t(x):
            # out = (x - v) * factor
            # out = tanh(out)
            # out = (out + yDelta) * scale
            return (tanh((x - v) * factor) + yDelta) * scale

        return t

    def plotFunction(self, func, folderName):
        # build data for function
        nPts = 201
        ptsGap = 5

        pts = linspace(0, 2, nPts).tolist()
        y = [round(func(x), 5) for x in pts]
        data = [[pts, y, 'bo']]
        pts = [pts[x] for x in range(0, nPts, ptsGap)]
        y = [y[k] for k in range(0, nPts, ptsGap)]
        data.append([pts, y, 'go'])

        # plot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for x, y, style in data:
            ax.plot(x, y, style)

        ax.set_xticks(pts)
        ax.set_yticks(y)
        ax.set_xlabel('bops/maxBops')
        ax.set_ylabel('Loss')
        ax.set_title('Bops ratio loss function')
        fig.set_size_inches(25, 10)

        fig.savefig('{}/bops_loss_func.png'.format(folderName))

    # structure =
    # @(acc_, x, scale, stretchFactor)
    #
    # (tanh((acc_ - x) * stretchFactor) * scale);
    #
    # acc = 1;
    # scale = 1;
    # stretchFactor = 50;
    # reward = 1.6;
    #
    # v = acc + (reward / (stretchFactor * scale));
    #
    # f = @(acc_)    ((structure(acc_, v, scale, stretchFactor) + 1) * 2.5);

    # def forward(self, input, target, model):
    #     bops_input = MaxBops = 0
    #
    #     alphasConvIdx = alphasDownsampleIdx = alphasLinearIdx = 0
    #
    #     for idx, l in enumerate(model.layersList):
    #         bops = []
    #         if isinstance(l, MixedConvWithReLU):
    #             alphas = model.alphasConv[alphasConvIdx]
    #             alphasConvIdx += 1
    #         elif isinstance(l, MixedLinear):
    #             alphas = model.alphasLinear[alphasLinearIdx]
    #             alphasLinearIdx += 1
    #         elif isinstance(l, MixedConv):
    #             alphas = model.alphasDownsample[alphasDownsampleIdx]
    #             alphasDownsampleIdx += 1
    #         _, uniform_bops = count_flops(self.uniform_model.layersList[idx].ops[0], self.batch_size, 1, l.in_planes)
    #         MaxBops += uniform_bops
    #
    #         for op in l.ops:
    #             bops_op = 'operation_{}_kernel_{}x{}_bit_{}_act_{}_channels_{}_residual_{}'.format(
    #                 str(type(l)).split(".")[-1], l.kernel_size, l.kernel_size, op.bitwidth, op.act_bitwidth,
    #                 l.in_planes, op.useResidual)
    #             bops.append(model.bopsDict[bops_op])
    #
    #         weights = F.softmax(alphas)
    #         bops_input += sum(a * op for a, op in zip(weights, bops))
    #
    #     # big penalization if bops over MaxBops
    #     penalization_factor = 1
    #     if (bops_input > MaxBops):
    #         penalization_factor = 5
    #     quant_loss = penalization_factor * (bops_input / MaxBops)
    #
    #     return self.search_loss(input, target) + (self.lmdba * quant_loss)

    # calculate BOPS per operation in each layer
#      self.bopsDict = {}
#      for l in self.layersList:
#          for op in l.ops:
#              bops_op = 'operation_{}_kernel_{}x{}_bit_{}_act_{}_channels_{}_residual_{}'. format(str(type(l)).split(".")[-1], l.kernel_size, l.kernel_size, op.bitwidth, op.act_bitwidth, l.in_planes,op.useResidual)
#              if bops_op not in self.bopsDict:
#                  _, curr_bops = count_flops(op, batch_size, 1, l.in_planes)
#                  self.bopsDict[bops_op] = curr_bops

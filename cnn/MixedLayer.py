from itertools import groupby
from abc import abstractmethod

from torch import cat, chunk, tensor, zeros, int32
from torch.nn import ModuleList, BatchNorm2d, Conv2d
from torch.distributions.multinomial import Multinomial
from torch.nn import functional as F

from cnn.MixedFilter import MixedFilter
from cnn.block import Block

from UNIQ.quantize import check_quantization
from NICE.quantize import ActQuant


# collects stats from forward output
def collectStats(type, val):
    funcs = [(lambda x: x.argmin(), lambda x: x.min()), (lambda x: '', lambda x: sum(x) / len(x)), (lambda x: x.argmax(), lambda x: x.max())]

    res = [[['Filter#', filterFunc(val)], ['Value', '{:.5f}'.format(valueFunc(val))]] for filterFunc, valueFunc in funcs]
    res = [type] + res

    return res


def postForward(self, _, output):
    assert (False)
    if self.quantized is True:
        # calc mean, max value per feature map to stats
        layerMax = tensor([output.select(1, j).max() for j in range(output.size(1))])
        layerAvg = tensor([(output.select(1, j).sum() / output.select(1, j).numel()) for j in range(output.size(1))])
        # save min, avg & max values for stats
        elements = [('Avg', layerAvg), ('Max', layerMax)]
        self.forwardStats = [collectStats(type, val) for type, val in elements]
        self.forwardStats.insert(0, ['Type', 'Min', 'Avg', 'Max'])

        # for i, m in enumerate(layerMax):
        #     if m.item() <= 1E-5:
        #         filter = self.filters[i]
        #         conv = filter.ops[0][filter.curr_alpha_idx].op[0].weight
        #         self.forwardStats.append([['Filter#', i], ['MaxVal', m], ['conv weights', conv]])

    else:
        self.forwardStats = None


# alphas = [[-0.77425, -0.51608, 1.99985, -1.40266], [0.45255, -0.58584, 1.79451, -2.35437], [0.31633, 0.15337, 1.02898, -2.19183],
#           [1.05364, -0.67785, 1.11637, -2.18531], [1.0003, -1.0216, -1.07731], [-0.05548, 0.604, 0.54426, -1.78592],
#           [0.40192, -0.2272, 1.0015, -1.86936], [0.77592, -1.54067, -0.33387], [0.62658, 0.45246, 0.14065, -1.91283]]
#
# alphaIdx = [0]
#
#
# def getAlphas():
#     res = tensor(alphas[alphaIdx[0]]).cuda()
#     alphaIdx[0] += 1
#     return res


class MixedLayer(Block):
    def __init__(self, nFilters, createMixedFilterFunc, useResidual=False):
        super(MixedLayer, self).__init__()

        # create mixed filters
        self.filters = ModuleList()
        for _ in range(nFilters):
            self.filters.append(createMixedFilterFunc())
        # make sure mixed filters are subclasses of MixedFilter
        assert (isinstance(self.filters[0], MixedFilter))

        # init operations alphas (weights)
        self.alphas = tensor((zeros(self.numOfOps())).cuda(), requires_grad=True)
        # self.alphas = tensor(getAlphas(), requires_grad=True)
        self.alphas = self.alphas.cuda()

        # # =========== change alphas distribution ==================
        # from math import log
        # filter = self.filters[0]
        # p = 13/16
        # logVal = p / (1 - p) * (self.numOfOps() - 1)
        # for i, op in enumerate(filter.opsList()):
        #     opBitwidth = op.getBitwidth()
        #     if opBitwidth == (3, 3) or opBitwidth == (3, None):
        #         self.alphas.data[i].fill_(log(logVal))

        # init filters current partition by alphas, i.e. how many filters are for each alpha, from each quantization
        self.currFiltersPartition = [0] * self.numOfOps()

        # # set filters distribution
        # if self.numOfOps() > 1:
        #     self.setAlphas([0.3125, 0.3125, 0.1875, 0.125, 0.0625])
        #     self.setFiltersPartition()

        # set forward function
        self.forwardFunc = self.residualForward if useResidual else self.standardForward

        # # register post forward hook
        # self.register_forward_hook(postForward)
        # self.forwardStats = None

        # set UNIQ parameters
        self.quantized = False
        self.added_noise = False

    def nFilters(self):
        return len(self.filters)

    def getLayers(self):
        return [self]

    def quantize(self, layerIdx):
        assert (self.added_noise is False)
        for op in self.opsList():
            assert (op.noise is False)
            assert (op.quant is False)
            op.quant = True

            op.quantizeFunc()
            assert (check_quantization(op.getModule(Conv2d).weight) <= (2 ** op.bitwidth[0]))
            # quantize activations during training
            for m in op.modules():
                if isinstance(m, ActQuant):
                    m.qunatize_during_training = True

        self.quantized = True
        print('quantized layer [{}] + quantize activations during training'.format(layerIdx))

    def unQuantize(self, layerIdx):
        assert (self.quantized is True)
        assert (self.added_noise is False)

        for op in self.opsList():
            assert (op.quant is True)
            op.quant = False
            op.restore_state()
            # remove activations quantization during training
            for m in op.modules():
                if isinstance(m, ActQuant):
                    m.qunatize_during_training = False

        self.quantized = False
        print('removed quantization in layer [{}] + removed activations quantization during training'.format(layerIdx))

    # just turn on op.noise flag
    # noise is being added in pre-forward hook
    def turnOnNoise(self, layerIdx):
        assert (self.quantized is False)
        for op in self.opsList():
            assert (op.noise is False)
            op.noise = True

        self.added_noise = True
        print('turned on noise in layer [{}]'.format(layerIdx))

    def turnOffNoise(self, layerIdx):
        assert (self.quantized is False)
        assert (self.added_noise is True)

        for op in self.opsList():
            assert (op.noise is True)
            op.noise = False

        self.added_noise = False
        print('turned off noise in layer [{}]'.format(layerIdx))

    # ratio is a list
    def setAlphas(self, ratio):
        self.alphas.data = tensor(ratio)

    # set filters curr_alpha_idx based on partition tensor
    # partition is IntTensor
    def setFiltersPartition(self, partition):
        assert (partition.sum().item() == self.nFilters())
        # reset current filters partition by alphas
        self.currFiltersPartition = [0] * self.numOfOps()
        # update filters curr_alpha_idx
        idx = 0
        for i, r in enumerate(partition):
            for _ in range(r):
                self.filters[idx].curr_alpha_idx = i
                self.currFiltersPartition[i] += 1
                idx += 1

    # set filters partition based on ratio
    # ratio is a tensor
    def __setFiltersPartitionFromRatio(self, ratio):
        # calc partition
        partition = (ratio * self.nFilters()).type(int32)
        # fix last ratio value to sum to nFilters
        if partition.sum().item() < self.nFilters():
            partition[-1] = self.nFilters() - partition[:-1].sum().item()

        self.setFiltersPartition(partition)

    # set filters partition based on alphas ratio
    def setFiltersPartitionByAlphas(self):
        probs = F.softmax(self.alphas, dim=-1)
        self.__setFiltersPartitionFromRatio(probs)

    def getCurrentFiltersPartition(self):
        return self.currFiltersPartition

    # input_bitwidth is a list of bitwidth per feature map
    def getBops(self, input_bitwidth):
        bops = 0.0
        # init bops map
        bopsMap = {}
        for f in self.filters:
            bops += f.getBops(input_bitwidth, bopsMap)

        return bops

    # returns filters current op bitwidth
    def getCurrentBitwidth(self):
        # collect filters current bitwidths
        bitwidths = [f.getCurrentBitwidth() for f in self.filters]
        # group bitwidths
        groups = groupby(bitwidths, lambda x: x)
        # create a list of tuples [bitwidth, number of filters]
        res = []
        for _, g in groups:
            g = list(g)
            res.append([g[0], len(g)])

        return res

    # create a list of layer output feature maps bitwidth
    def getCurrentOutputBitwidth(self):
        outputBitwidth = [f.getCurrentOutputBitwidth() for f in self.filters]
        return outputBitwidth

    def opsList(self):
        for filter in self.filters:
            for op in filter.opsList():
                yield op

    def getAllBitwidths(self):
        # it doesn't matter which filter we take, the attributes are the same in all filters
        return self.filters[0].getAllBitwidths()

    def numOfOps(self):
        # it doesn't matter which filter we take, the attributes are the same in all filters
        return self.filters[0].numOfOps()

    def outputLayer(self):
        return self

    # select alpha based on alphas distribution
    def choosePathByAlphas(self):
        dist = Multinomial(total_count=self.nFilters(), logits=self.alphas)
        partition = dist.sample().type(int32)
        self.setFiltersPartition(partition)

    @abstractmethod
    def preResidualForward(self, x):
        raise NotImplementedError('subclasses must override preResidualForward()!')

    # operations to perform after adding residual
    def postResidualForward(self, x):
        out = x
        # apply ReLU if exists
        if self.filters[0].postResidualForward:
            out = []
            # split out1 to chunks again
            x = chunk(x, self.nFilters(), dim=1)
            # apply selected op in each filter
            for i, f in enumerate(self.filters):
                res = f.postResidualForward(x[i])
                out.append(res)
            # concat filters output
            out = cat(out, 1)

        return out

    def forward(self, x):
        return self.forwardFunc(self, x)

    # standard forward
    @staticmethod
    def standardForward(layer, x):
        out = layer.preResidualForward(x)
        out = layer.postResidualForward(out)

        return out

    # forward with residual
    @staticmethod
    def residualForward(layer, input):
        x, residual = input
        out = layer.preResidualForward(x)
        # add residual
        out += residual
        out = layer.postResidualForward(out)

        return out


class MixedLayerNoBN(MixedLayer):
    def __init__(self, nFilters, createMixedFilterFunc, useResidual=False):
        super(MixedLayerNoBN, self).__init__(nFilters, createMixedFilterFunc, useResidual)

    # operations to perform before adding residual
    def preResidualForward(self, x):
        out = []
        # apply selected op in each filter
        for f in self.filters:
            res = f(x)
            out.append(res)
        # concat filters output
        out = cat(out, 1)

        return out


class MixedLayerWithBN(MixedLayer):
    def __init__(self, nFilters, createMixedFilterFunc, useResidual=False):
        super(MixedLayerWithBN, self).__init__(nFilters, createMixedFilterFunc, useResidual)

        # init batch norm
        self.bn = BatchNorm2d(nFilters)

    # perform the convolution operation
    def forwardConv(self, x):
        out = []
        # apply selected op in each filter
        for f in self.filters:
            res = f(x)
            out.append(res)
        # concat filters output
        out = cat(out, 1)

        return out

    # operations to perform before adding residual
    def preResidualForward(self, x):
        out = self.forwardConv(x)
        # apply batch norm
        out = self.bn(out)

        return out

# bitwidth list is the same for all filters, therefore we can use the 1st filter list
# def getOutputBitwidthList(self):
#     return self.filters[0].getOutputBitwidthList()

# def evalMode(self):
#     pass

# # select random alpha
# def chooseRandomPath(self):
#     pass

# # quantize activations during training
# def quantActOnTraining(self, layerIdx):
#     assert (self.quantized is True)
#     assert (self.added_noise is False)
#
#     for op in self.opsList:
#         for m in op.modules():
#             if isinstance(m, ActQuant):
#                 m.qunatize_during_training = True
#
#     print('turned on qunatize_during_training in layer [{}]'.format(layerIdx))
#
# # stop quantize activations during training
# def turnOnGradients(self, layerIdx):
#     assert (self.quantized is False)
#     assert (self.added_noise is False)
#
#     for op in self.opsList:
#         for m in op.modules():
#             if isinstance(m, ActQuant):
#                 m.qunatize_during_training = False
#
#     print('turned off qunatize_during_training in layer [{}]'.format(layerIdx))

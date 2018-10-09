from torch import cat, chunk, tensor, ones, IntTensor
from torch.nn import ModuleList, BatchNorm2d
from torch.distributions.multinomial import Multinomial

from cnn.MixedFilter import MixedFilter
from cnn.block import Block


class MixedLayer(Block):
    def __init__(self, nFilters, createMixedFilterFunc, useResidual=False):
        super(MixedLayer, self).__init__()

        # create mixed filters
        self.filters = ModuleList()
        for _ in range(nFilters):
            self.filters.append(createMixedFilterFunc())
        # make sure mixed filters are subclasses of MixedFilter
        assert (isinstance(self.filters[0], MixedFilter))

        # init batch norm
        self.bn = BatchNorm2d(nFilters)

        # init operations alphas (weights)
        value = 1.0 / self.numOfOps()
        self.alphas = tensor((ones(self.numOfOps()) * value).cuda(), requires_grad=True)

        # set filters distribution
        if self.numOfOps() > 1:
            self.setAlphas([0.3125, 0.3125, 0.1875, 0.125, 0.0625])
            # self.setFiltersPartition()

        # init list of all operations (including copies) as single long list
        # for cases we have to modify all ops
        self.opsList = []
        for filter in self.filters:
            self.opsList.extend(filter.getOps())

        # set forward function
        if useResidual:
            self.forward = self.residualForward

    def nFilters(self):
        return len(self.filters)

    # ratio is a list
    def setAlphas(self, ratio):
        self.alphas.data = tensor(ratio)

    # set filters curr_alpha_idx based on partition tensor
    # partition is IntTensor
    def __setFiltersPartition(self, partition):
        assert (partition.sum().item() == self.nFilters())
        # update filters curr_alpha_idx
        idx = 0
        for i, r in enumerate(partition):
            for _ in range(r):
                self.filters[idx].curr_alpha_idx = i
                idx += 1

    # set filters partition based on ratio
    # ratio is a tensor
    def __setFiltersPartitionFromRatio(self, ratio):
        # calc partition
        partition = (ratio * self.nFilters()).type(IntTensor)
        # fix last ratio value to sum to nFilters
        if partition.sum().item() < self.nFilters():
            partition[-1] = self.nFilters() - partition[:-1].sum().item()

        self.__setFiltersPartition(partition)

    # set filters partition based on alphas ratio
    def setFiltersPartition(self):
        self.__setFiltersPartitionFromRatio(self.alphas)

    # perform the convolution operation
    def forwardConv(self, x):
        out = []
        # apply selected op in each filter
        for f in self.filters:
            res = f.forwardConv(x)
            out.append(res)
        # concat filters output
        out = cat(out, 1)

        return out

    # perform the ReLU operation
    def forwardReLU(self, x):
        out = []
        # apply selected op in each filter
        for i, f in enumerate(self.filters):
            res = f.forwardReLU(x[i])
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

    # operations to perform after adding residual
    def postResidualForward(self, out):
        # apply ReLU if exists
        if self.filters[0].forwardReLU:
            # split out1 to chunks again
            out = chunk(out, self.nFilters(), dim=1)
            out = self.forwardReLU(out)

        return out

    # standard forward
    def forward(self, x):
        out = self.preResidualForward(x)
        out = self.postResidualForward(out)

        return out

    # forward with residual
    def residualForward(self, x, residual):
        out = self.preResidualForward(x)
        # add residual
        out += residual
        out = self.postResidualForward(out)

        return out

    # input_bitwidth is a list of bitwidth per feature map
    def getBops(self, input_bitwidth):
        bops = 0.0
        for f in self.filters:
            bops += f.getBops(input_bitwidth)

        return bops

    # create a list of layer output feature maps bitwidth
    def getCurrentOutputBitwidth(self):
        outputBitwidth = [f.getCurrentOutputBitwidth() for f in self.filters]
        return outputBitwidth

    def getOps(self):
        return self.opsList

    def getAllBitwidths(self):
        # it doesn't matter which filter we take, the attributes are the same in all filters
        return self.filters[0].getAllBitwidths()

    def numOfOps(self):
        # it doesn't matter which filter we take, the attributes are the same in all filters
        return self.filters[0].numOfOps()

    def outputLayer(self):
        return self

    ## functions that need to be examined about their correctness
    # bitwidth list is the same for all filters, therefore we can use the 1st filter list
    def getOutputBitwidthList(self):
        return self.filters[0].getOutputBitwidthList()

    # # select random alpha
    # def chooseRandomPath(self):
    #     pass

    # select alpha based on alphas distribution
    def choosePathByAlphas(self):
        dist = Multinomial(total_count=self.nFilters(), probs=self.alphas)
        partition = dist.sample().type(IntTensor)
        self.__setFiltersPartition(partition)

    def evalMode(self):
        pass

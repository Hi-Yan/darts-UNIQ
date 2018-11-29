from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, Module

from .ResNet import ResNet, BasicBlock


class BasicBlockFullPrecision(Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(BasicBlockFullPrecision, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        self.block1 = Sequential(
            Conv2d(in_planes, out_planes, kernel_size, stride=stride1, padding=1, bias=False),
            BatchNorm2d(out_planes),
            ReLU(inplace=True)
        )

        self.block2 = Sequential(
            Conv2d(out_planes, out_planes, kernel_size, stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes)
        )

        self.relu = ReLU(inplace=True)

        self.downsample = Sequential(
            Conv2d(in_planes, out_planes, kernel_size=1, stride=stride1, bias=False),
            BatchNorm2d(out_planes)
        ) if in_planes != out_planes else None

        # self.out_planes = out_planes
        # self.bitwidth = 8
        # # build Conv2d blocks bops dummies for bops calculation
        # bopsBlock1 = Conv2d(in_planes, 1, kernel_size, stride=stride1, padding=1, bias=False)
        # bopsBlock2 = Conv2d(out_planes, 1, kernel_size, stride=stride, padding=1, bias=False)
        # bopsDownsample = Conv2d(in_planes, 1, kernel_size=1, stride=stride1, bias=False) if in_planes != out_planes else None
        # # count bops
        # self.bops = {
        #     self.block1: count_flops(bopsBlock1, input_size[0], in_planes),
        #     self.block2: count_flops(bopsBlock2, input_size[-1], out_planes),
        # }
        # if bopsDownsample:
        #     self.bops[self.downsample] = count_flops(bopsDownsample, input_size[0], in_planes)

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.block1(x)
        out = self.block2(out)
        out += residual
        out = self.relu(out)

        return out

    # # input_bitwidth is a list of bitwidth per feature map
    # def getBops(self, input_bitwidth):
    #     bops = self.countOpsBops(self.bops[self.block1], input_bitwidth)
    #     if self.downsample:
    #         bops += self.countOpsBops(self.bops[self.downsample], input_bitwidth)
    #
    #     input_bitwidth = [self.bitwidth] * self.out_planes
    #     bops += self.countOpsBops(self.bops[self.block2], input_bitwidth)
    #
    #     output_bitwidth = [self.bitwidth] * self.out_planes
    #     # we calculated bops for a single filter, need to multiply by number of filters
    #     bops *= self.out_planes
    #
    #     return bops, output_bitwidth
    #
    # def countOpsBops(self, opBopsData, input_bitwidth):
    #     # set block filter's bitwidth
    #     bitwidth = self.bitwidth
    #     # get bops values
    #     mults, adds, calcMacObj, batch_size = opBopsData
    #     # calc max_mac_value
    #     max_mac_value = 0
    #     for act_bitwidth in input_bitwidth:
    #         max_mac_value += calcMacObj.calc(bitwidth, act_bitwidth)
    #     # init log2_max_mac_value
    #     log2_max_mac_value = ceil(log2(max_mac_value))
    #     # calc bops
    #     bops = 0.0
    #     # calc bops mults
    #     for act_bitwidth in input_bitwidth:
    #         bops += mults * (bitwidth - 1) * act_bitwidth
    #     # add bops adds
    #     bops += adds * log2_max_mac_value
    #     # divide by batch size
    #     bops /= batch_size
    #
    #     return bops


class ThinResNet(ResNet):
    def __init__(self, args):
        super(ThinResNet, self).__init__(args)

    # init layers (type, in_planes, out_planes)
    def initLayersPlanes(self):
        return [(self.createMixedLayer, 3, 16, 32), (BasicBlock, 16, 16, [32]),
                (BasicBlock, 16, 32, [32, 16]), (BasicBlock, 32, 64, [16, 8])]

    def buildStateDictMap(self, chckpntDict):
        map = {}
        map['conv1'] = 'layers.0.ops.0.0.op.0'
        map['bn1'] = 'layers.0.ops.0.0.op.1'
        map['relu'] = 'layers.0.ops.0.0.op.2'

        layersNumberMap = [(1, 0, 1)]
        for n1, n2, m in layersNumberMap:
            map['layer{}.{}.conv1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.0'.format(m)
            map['layer{}.{}.bn1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.1'.format(m)
            map['layer{}.{}.relu1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.2'.format(m)
            map['layer{}.{}.conv2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.0'.format(m)
            map['layer{}.{}.bn2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.1'.format(m)
            map['layer{}.{}.relu2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.2'.format(m)

        downsampleLayersMap = [(2, 0, 2), (3, 0, 3)]
        for n1, n2, m in downsampleLayersMap:
            map['layer{}.{}.conv1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.0'.format(m)
            map['layer{}.{}.bn1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.1'.format(m)
            map['layer{}.{}.relu1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.2'.format(m)
            map['layer{}.{}.conv2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.0'.format(m)
            map['layer{}.{}.bn2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.1'.format(m)
            map['layer{}.{}.relu2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.2'.format(m)
            map['layer{}.{}.downsample.0'.format(n1, n2)] = 'layers.{}.downsample.ops.0.0.op.0'.format(m)
            map['layer{}.{}.downsample.1'.format(n1, n2)] = 'layers.{}.downsample.ops.0.0.op.1'.format(m)

        map['fc'] = 'fc'

        return map

# def buildStateDictMap(self, chckpntDict):
#     map = {}
#     map['conv1'] = 'layers.0.ops.0.0.op.0'
#     map['bn1'] = 'layers.0.bn'
#     map['relu'] = 'layers.0.ops.0.0.op.1'
#
#     layersNumberMap = [(1, 0, 1)]
#     for n1, n2, m in layersNumberMap:
#         map['layer{}.{}.conv1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.0'.format(m)
#         map['layer{}.{}.bn1'.format(n1, n2)] = 'layers.{}.block1.bn'.format(m)
#         map['layer{}.{}.relu1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.1'.format(m)
#         map['layer{}.{}.conv2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.0'.format(m)
#         map['layer{}.{}.bn2'.format(n1, n2)] = 'layers.{}.block2.bn'.format(m)
#         map['layer{}.{}.relu2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.1'.format(m)
#
#     downsampleLayersMap = [(2, 0, 2), (3, 0, 3)]
#     for n1, n2, m in downsampleLayersMap:
#         map['layer{}.{}.conv1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.0'.format(m)
#         map['layer{}.{}.bn1'.format(n1, n2)] = 'layers.{}.block1.bn'.format(m)
#         map['layer{}.{}.relu1'.format(n1, n2)] = 'layers.{}.block1.ops.0.0.op.1'.format(m)
#         map['layer{}.{}.conv2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.0'.format(m)
#         map['layer{}.{}.bn2'.format(n1, n2)] = 'layers.{}.block2.bn'.format(m)
#         map['layer{}.{}.relu2'.format(n1, n2)] = 'layers.{}.block2.ops.0.0.op.1'.format(m)
#         map['layer{}.{}.downsample.0'.format(n1, n2)] = 'layers.{}.downsample.ops.0.0.op.0'.format(m)
#         map['layer{}.{}.downsample.1'.format(n1, n2)] = 'layers.{}.downsample.bn'.format(m)
#
#     map['fc'] = 'fc'
#
#     return map

# def buildStateDictMap(self, chckpntDict):
#     def iterateKey(chckpntDict, map, key1, key2, dstKey):
#         keyIdx = 0
#         key = '{}.{}.{}'.format(key1, key2, keyIdx)
#         while '{}.weight'.format(key) in chckpntDict:
#             map[key] = dstKey.format(key2, keyIdx)
#             keyIdx += 1
#             key = '{}.{}.{}'.format(key1, key2, keyIdx)
#
#     def getNextblock(obj, key, blockNum):
#         block = getattr(obj, key, None)
#         return block, (blockNum + 1)
#
#     # =============================================================================
#     map = {
#         'block1.0': 'layers.0.ops.0.0.op.0.0',
#         'block1.1': 'layers.0.ops.0.0.op.0.1',
#         'fc': 'fc'
#     }
#
#     for i, layer in enumerate(self.layers[1:]):
#         dstPrefix = 'layers.{}.'.format(i + 1)
#         key1 = 'block{}'.format(i + 2)
#         innerBlockNum = 1
#         key2 = 'block{}'.format(innerBlockNum)
#         c, innerBlockNum = getNextblock(layer, key2, innerBlockNum)
#         while c:
#             iterateKey(chckpntDict, map, key1, key2, dstPrefix + '{}.ops.0.0.op.0.{}')
#             key2 = 'block{}'.format(innerBlockNum)
#             c, innerBlockNum = getNextblock(layer, key2, innerBlockNum)
#
#         # copy downsample
#         key2 = 'downsample'
#         c, innerBlockNum = getNextblock(layer, key2, innerBlockNum)
#         if c:
#             iterateKey(chckpntDict, map, key1, key2, dstPrefix + '{}.ops.0.0.op.{}')
#
#     return map

# # ====================================================
# # for training pre_trained, i.e. full precision
# # ====================================================
# class ThinResNet(Module):
#     def __init__(self, args):
#         super(ThinResNet, self).__init__()
#
#         self.block1 = Sequential(Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), BatchNorm2d(16), ReLU(inplace=True))
#
#         ## ThinResNet =======================================================
#         layers = [
#             BasicBlockFullPrecision(16, 16, kernel_size=3, stride=1),
#             BasicBlockFullPrecision(16, 32, kernel_size=3, stride=1),
#             BasicBlockFullPrecision(32, 64, kernel_size=3, stride=1)
#         ]
#
#         ## ResNet =======================================================
#         # layers = [
#         #     BasicBlockFullPrecision(16, 16, kernel_size=3, stride=1),
#         #     BasicBlockFullPrecision(16, 16, kernel_size=3, stride=1),
#         #     BasicBlockFullPrecision(16, 16, kernel_size=3, stride=1),
#         #     BasicBlockFullPrecision(16, 32, kernel_size=3, stride=1),
#         #     BasicBlockFullPrecision(32, 32, kernel_size=3, stride=1),
#         #     BasicBlockFullPrecision(32, 32, kernel_size=3, stride=1),
#         #     BasicBlockFullPrecision(32, 64, kernel_size=3, stride=1),
#         #     BasicBlockFullPrecision(64, 64, kernel_size=3, stride=1),
#         #     BasicBlockFullPrecision(64, 64, kernel_size=3, stride=1)
#         # ]
#
#         self._nLayers = len(layers) * 2
#
#         i = 2
#         for l in layers:
#             setattr(self, 'block{}'.format(i), l)
#             i += 1
#
#         from torch.nn.modules import AvgPool2d, Linear
#         self.avgpool = AvgPool2d(8)
#         self.fc = Linear(64, args.nClasses)
#
#         self.nPerms = 7
#         self.learnable_params = []
#         self.layersList = []
#         self.nLayersQuantCompleted = self.nLayers()
#
#     def forward(self, x):
#         blockIdx = 1
#         out = x
#
#         block = getattr(self, 'block{}'.format(blockIdx), None)
#         while block is not None:
#             out = block(out)
#             blockIdx += 1
#             block = getattr(self, 'block{}'.format(blockIdx), None)
#
#         out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#
#         return out
#
#     def loadPreTrained(self, path, logger, gpu):
#         return False
#
#     def topOps(self, k):
#         return []
#
#     def nLayers(self):
#         return self._nLayers
#
#     def switch_stage(self, loggerFuncs=None):
#         pass
#
#     def choosePathByAlphas(self):
#         pass
#
#     def save_alphas_state(self):
#         return []
#
#     def calcBopsRatio(self):
#         return 1.2

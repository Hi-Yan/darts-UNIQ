from abc import abstractmethod

from pandas import DataFrame
from os.path import exists

from torch import ones, zeros, no_grad
from torch.nn import Module, CrossEntropyLoss
from torch.nn import functional as F
from torch import load as loadModel
from torch import save as saveModel

from cnn.MixedLayer import MixedLayer
from cnn.uniq_loss import UniqLoss
from cnn.statistics import Statistics


# from UNIQ.quantize import backup_weights, restore_weights, quantize

# def save_quant_state(self, _):
#     assert (False)
#     assert (self.noise is False)
#     if self.quant and not self.noise and self.training:
#         self.full_parameters = {}
#         layers_list = self.get_layers_list()
#         layers_steps = self.get_layers_steps(layers_list)
#         assert (len(layers_steps) == 1)
#
#         self.full_parameters = backup_weights(layers_steps[0], self.full_parameters)
#         quantize(layers_steps[0], bitwidth=self.bitwidth[0])
#
#
# def restore_quant_state(self, _, __):
#     assert (False)
#     assert (self.noise is False)
#     if self.quant and not self.noise and self.training:
#         layers_list = self.get_layers_list()
#         layers_steps = self.get_layers_steps(layers_list)
#         assert (len(layers_steps) == 1)
#
#         restore_weights(layers_steps[0], self.full_parameters)  # Restore the quantized layers


class BaseNet(Module):
    # init bitwidth of input to model
    modelInputBitwidth = 8
    modelInputnFeatureMaps = 3

    # counts the entire model bops in discrete mode
    def countBopsDiscrete(self):
        totalBops = 0
        # input_bitwidth is a list of bitwidth per feature map
        input_bitwidth = [self.modelInputBitwidth] * self.modelInputnFeatureMaps

        for layer in self.layers:
            totalBops += layer.getBops(input_bitwidth)
            input_bitwidth = layer.getCurrentOutputBitwidth()

        return totalBops

    def countBops(self):
        # wrapper is needed because countBopsFuncs is defined outside __init__()
        return self.countBopsFunc(self)

    countBopsFuncs = dict(discrete=countBopsDiscrete)

    alphasCsvFileName = 'alphas.csv'

    def __init__(self, args, initLayersParams):
        super(BaseNet, self).__init__()
        # init save folder
        saveFolder = args.save
        # init layers
        self.layers = self.initLayers(initLayersParams)
        # build mixture layers list
        self.layersList = [m for m in self.modules() if isinstance(m, MixedLayer)]
        # set bops counter function
        self.countBopsFunc = self.countBopsFuncs[args.bopsCounter]
        # init criterion
        if 'baselineBops' not in args:
            args.baselineBops = self.countBops()
        self._criterion = UniqLoss(args)
        self._criterion = self._criterion.cuda()
        # init statistics
        self.stats = Statistics(self.layersList, saveFolder)
        # collect learnable params (weights)
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]
        # init learnable alphas
        self.learnable_alphas = self.getLearnableAlphas()
        # init number of layers we have completed its quantization
        self.nLayersQuantCompleted = 0
        # init layers permutation list
        self.layersPerm = []
        # init number of permutations counter
        self.nPerms = 1
        for layer in self.layersList:
            # add layer numOps range to permutation list
            self.layersPerm.append(list(range(len(layer.alphas))))
            self.nPerms *= len(layer.alphas)

        # init alphas DataFrame
        self.alphas_df = None
        self.__initAlphasDataFrame(saveFolder)

    @abstractmethod
    def initLayers(self, params):
        raise NotImplementedError('subclasses must override initLayers()!')

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('subclasses must override forward()!')

    @abstractmethod
    def switch_stage(self, logger=None):
        raise NotImplementedError('subclasses must override switch_stage()!')

    @abstractmethod
    def loadUNIQPreTrained(self, checkpoint):
        raise NotImplementedError('subclasses must override loadUNIQPreTrained()!')

    @abstractmethod
    def loadSingleOpPreTrained(self, checkpoint):
        raise NotImplementedError('subclasses must override loadSingleOpPreTrained()!')

    @abstractmethod
    def turnOnWeights(self):
        raise NotImplementedError('subclasses must override turnOnWeights()!')

    def nLayers(self):
        return len(self.layersList)

    def getLearnableAlphas(self):
        return [layer.alphas for layer in self.layersList if layer.alphas.requires_grad is True]

    def updateLearnableAlphas(self):
        self.learnable_alphas = self.getLearnableAlphas()

    def arch_parameters(self):
        return self.learnable_alphas

    # def __loadStatistics(self, filename):
    #     if exists(filename):
    #         # stats is a list of dicts per layer
    #         stats = loadModel(filename)
    #         print('Loading statistics')
    #
    #         for i, layer in enumerate(self.layersList):
    #             # get layer dict
    #             layerStats = stats[i]
    #             # iterate over layer filters
    #             for filter in layer.filters:
    #                 # iterate over filter modules
    #                 for m in filter.modules():
    #                     # create module type as string
    #                     moduleType = '{}'.format(type(m))
    #                     NICEprefix = "'NICE."
    #                     if NICEprefix in moduleType:
    #                         moduleType = moduleType.replace(NICEprefix, "'")
    #
    #                     # check if type string is in dict
    #                     if moduleType in layerStats:
    #                         # go over dict keys, which is the module variables
    #                         for varName in layerStats[moduleType].keys():
    #                             v = getattr(m, varName)
    #                             # if variable has value in dict, assign it
    #                             if v is not None:
    #                                 v.data = layerStats[moduleType][varName].data

    def calcStatistics(self, statistics_queue):
        # prepare for collecting statistics, reset register_buffers values
        for layer in self.layersList:
            for op in layer.opsList:
                conv = op.op[0]
                # reset conv register_buffer values
                conv.layer_b = ones(1).cuda()
                conv.layer_basis = ones(1).cuda()
                conv.initial_clamp_value = ones(1).cuda()
                # get actquant
                actQuant = op.op[1] if len(op.op) > 1 else None
                if actQuant:
                    # reset actquant register_buffer values
                    actQuant.running_mean = zeros(1).cuda()
                    actQuant.running_std = zeros(1).cuda()
                    actQuant.clamp_val.data = zeros(1).cuda()
                    # set actquant to statistics forward
                    actQuant.forward = actQuant.statisticsForward

        # # turn off noise in 1st layer, collecting statistics is performed in full-precision
        # if len(self.layersList) > 0:
        #     layerIdx = 0
        #     self.layersList[layerIdx].turnOffNoise(layerIdx)

        # train for statistics
        criterion = CrossEntropyLoss().cuda()
        nBatches = 80
        self.eval()
        with no_grad():
            for step, (input, target) in enumerate(statistics_queue):
                if step >= nBatches:
                    break

                output = self(input.cuda())
                criterion(output, target.cuda())
        # apply quantize class statistics functions
        for layer in self.layersList:
            for op in layer.opsList:
                opModulesList = list(op.modules())
                op.quantize.get_act_max_value_from_pre_calc_stats(opModulesList)
                op.quantize.set_weight_basis(opModulesList, None)
                # restore actquant forward function
                actQuant = op.op[1] if len(op.op) > 1 else None
                # set actquant to standard forward
                if actQuant:
                    actQuant.forward = actQuant.standardForward

        # # turn on noise in 1st layer, collecting statistics is performed in full-precision
        # if len(self.layersList) > 0:
        #     layerIdx = 0
        #     self.layersList[layerIdx].turnOnNoise(layerIdx)

    # updates statistics in checkpoint, in order to avoid calculating statistics when loading model from checkpoint
    def updateCheckpointStatistics(self, checkpoint, path, statistics_queue):
        needToUpdate = ('updated_statistics' not in checkpoint) or (checkpoint['updated_statistics'] is not True)
        if needToUpdate:
            # load checkpoint weights
            self.load_state_dict(checkpoint['state_dict'])
            # calc weights statistics
            self.calcStatistics(statistics_queue)
            # update checkpoint
            checkpoint['state_dict'] = self.state_dict()
            checkpoint['updated_statistics'] = True
            # save updated checkpoint
            saveModel(checkpoint, path)

        return needToUpdate

    # layer_basis is a function of filter quantization,
    # therefore we have to update its value bases on weight_max_int, which is a function of weights bitwidth
    def __updateStatistics(self):
        for layer in self.layersList:
            for op in layer.opsList:
                conv = op.op[0]
                # update layer_basis value based on weights bitwidth
                conv.layer_basis = conv.initial_clamp_value / op.quantize.weight_max_int

    def loadPreTrained(self, path, logger, gpu):
        # init bool flag whether we loaded ops in the same layer with equal or different weights
        loadOpsWithDifferentWeights = False
        loggerRows = []
        if path is not None:
            if exists(path):
                # load checkpoint
                checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
                assert (checkpoint['updated_statistics'] is True)
                chckpntStateDict = checkpoint['state_dict']
                # load model state dict keys
                modelStateDictKeys = set(self.state_dict().keys())
                # compare dictionaries
                dictDiff = modelStateDictKeys.symmetric_difference(set(chckpntStateDict.keys()))
                # update flag value
                loadOpsWithDifferentWeights = len(dictDiff) == 0
                # decide how to load checkpoint state dict
                if loadOpsWithDifferentWeights:
                    # load directly, keys are the same
                    self.load_state_dict(chckpntStateDict)
                else:
                    # use some function to map keys
                    loadFuncs = [self.loadUNIQPreTrained, self.loadSingleOpPreTrained]
                    for func in loadFuncs:
                        loadSuccess = func(chckpntStateDict)
                        if loadSuccess is not False:
                            break

                if loadSuccess is not False:
                    # add info rows about checkpoint
                    loggerRows.append(['Path', '{}'.format(path)])
                    loggerRows.append(['Validation accuracy', '{:.5f}'.format(checkpoint['best_prec1'])])
                    loggerRows.append(['updated_statistics', checkpoint['updated_statistics']])
                    # update statistics
                    self.__updateStatistics()
                    # check if model includes stats
                    modelIncludesStats = False
                    for key in chckpntStateDict.keys():
                        if key.endswith('.layer_basis'):
                            modelIncludesStats = True
                            break
                    loggerRows.append(['Includes stats', '{}'.format(modelIncludesStats)])
                else:
                    loggerRows.append(['Path', 'Failed to load pre-trained from [{}], state_dict does not fit'.format(path)])
            else:
                loggerRows.append(['Path', 'Failed to load pre-trained from [{}], path does not exists'.format(path)])

            # load pre-trained model if we tried to load pre-trained
            logger.addInfoTable('Pre-trained model', loggerRows)

        return loadOpsWithDifferentWeights

    def __initAlphasDataFrame(self, saveFolder):
        if saveFolder:
            # update save path if saveFolder exists
            self.alphasCsvFileName = '{}/{}'.format(saveFolder, self.alphasCsvFileName)
            # init DataFrame cols
            cols = ['Epoch', 'Batch']
            cols += ['Layer_{}'.format(i) for i in range(self.nLayers())]
            self.cols = cols
            # init DataFrame
            self.alphas_df = DataFrame([], columns=cols)
            # set init data
            data = ['init', 'init']
            # save alphas data
            self.save_alphas_to_csv(data)

    def turnOffAlphas(self):
        for layer in self.layersList:
            layer.alphas.grad = None

    def calcBopsRatio(self):
        return self._criterion.calcBopsRatio(self.countBops())

    def choosePathByAlphas(self):
        for l in self.layers:
            l.choosePathByAlphas()

    # set curr_alpha_idx to each filter by alphas values
    def setFiltersByAlphas(self):
        for layer in self.layersList:
            layer.setFiltersPartitionByAlphas()

    def isQuantized(self):
        for layer in self.layersList:
            assert (layer.quantized is True)
            assert (layer.added_noise is False)

        return True

    def resetForwardCounters(self):
        for layer in self.layersList:
            for filter in layer.filters:
                # reset filter counters
                filter.resetOpsForwardCounters()

    # return top k operations per layer
    def topOps(self, k):
        top = []
        for layer in self.layersList:
            # calc weights from alphas and sort them
            weights = F.softmax(layer.alphas, dim=-1)
            wSorted, wIndices = weights.sort(descending=True)
            # keep only top-k
            wSorted = wSorted[:k]
            wIndices = wIndices[:k]
            # add to top
            top.append([(i, w.item(), layer.alphas[i], layer.filters[0].ops[0][i]) for w, i in zip(wSorted, wIndices)])

        return top

    # save alphas values to csv
    def save_alphas_to_csv(self, data):
        if self.alphas_df is not None:
            data += [[round(e.item(), 5) for e in layer.alphas] for layer in self.layersList]
            # create new row
            d = DataFrame([data], columns=self.cols)
            # add row
            self.alphas_df = self.alphas_df.append(d)
            # save DataFrame
            self.alphas_df.to_csv(self.alphasCsvFileName)

    # create list of tuples (layer index, layer alphas)
    def save_alphas_state(self):
        return [(i, layer.alphas) for i, layer in enumerate(self.layersList)]

    def load_alphas_state(self, state):
        for layerIdx, alphas in state:
            self.layersList[layerIdx].alphas.data = alphas

# # select random alpha
# def chooseRandomPath(self):
#     for l in self.layers:
#         l.chooseRandomPath()

# # layerIdx, alphaIdx meaning: self.layersList[layerIdx].curr_alpha_idx = alphaIdx
# # def choosePathByAlphas(self, layerIdx=None, alphaIdx=None):
# def choosePathByAlphas(self):
#     for l in self.layers:
#         l.choosePathByAlphas()
#
#     if (layerIdx is not None) and (alphaIdx is not None):
#         layer = self.layersList[layerIdx]
#         layer.curr_alpha_idx = alphaIdx

# def evalMode(self):
#     for l in self.layers:
#         l.evalMode()
#
#     # calc bops ratio
#     return self.calcBopsRatio()

# def uniformMode(self):
#     for l in self.layersList:
#         l.uniformMode(self._criterion.baselineBits)
#
#     # calc bops ratio
#     return self.calcBopsRatio()

# def turnOffAlphas(self):
#     for layer in self.layersList:
#         # turn off alphas gradients
#         layer.alphas.requires_grad = False
#
#     self.learnable_alphas = []

# def turnOnAlphas(self):
#     self.learnable_alphas = []
#     for layer in self.layersList:
#         # turn on alphas gradients
#         layer.alphas.requires_grad = True
#         self.learnable_alphas.append(layer.alphas)
#
#         for op in layer.getOps():
#             # turn off noise in op
#             op.noise = False
#
#             ## ==== for tinyNet ====
#             # # set pre & post quantization hooks, from now on we want to quantize these ops
#             # op.register_forward_pre_hook(save_quant_state)
#             # op.register_forward_hook(restore_quant_state)


# # convert current model to discrete, i.e. keep nOpsPerLayer optimal operations per layer
# def toDiscrete(self, nOpsPerLayer=1):
#     for layer in self.layersList:
#         # calc weights from alphas and sort them
#         weights = F.softmax(layer.alphas, dim=-1)
#         _, wIndices = weights.sort(descending=True)
#         # update layer alphas
#         layer.alphas = layer.alphas[wIndices[:nOpsPerLayer]]
#         #        layer.alphas = tensor(tensor(layer.alphas.tolist()).cuda(), requires_grad=True)
#         layer.alphas = tensor(tensor(layer.alphas.tolist()).cuda())
#         # take indices of ops we want to remove from layer
#         wIndices = wIndices[nOpsPerLayer:]
#         # convert to list
#         wIndices = wIndices.tolist()
#         # sort indices ascending
#         wIndices.sort()
#         # remove ops and corresponding bops from layer
#         for w in reversed(wIndices):
#             del layer.ops[w]
#             del layer.bops[w]

# def loadBitwidthWeigths(self, stateDict, MaxBopsBits, bitwidth):
#     # check idx of MaxBopsBits inside bitwidths
#     maxBopsBitsIdx = bitwidth.index(MaxBopsBits)
#     maxBopsStateDict = OrderedDict()
#     opsKey = 'ops.'
#     for key in stateDict.keys():
#         # if operation is for max bops bits idx
#         if opsKey in key:
#             keyOp_num = key.split(opsKey)[1][0]
#             if int(keyOp_num) == maxBopsBitsIdx:
#                 maxBopsKey = key.replace(opsKey + keyOp_num, opsKey + '0')
#                 maxBopsStateDict[maxBopsKey] = stateDict[key]
#         else:
#             maxBopsStateDict[key] = stateDict[key]
#
#     self.load_state_dict(maxBopsStateDict)

# def _loss(self, input, target):
#     totalLoss = 0.0
#     nIter = min(self.nPerms, 1000)
#     for _ in range(nIter):
#         logits = self.forward(input)
#
#         # calc alphas product
#         alphasProduct = 1.0
#         for layer in self.layersList:
#             probs = F.softmax(layer.alphas)
#             alphasProduct *= probs[layer.curr_alpha_idx]
#
#         permLoss = alphasProduct * self._criterion(logits, target, self.countBops())
#         # permLoss = self._criterion(logits, target, self.countBops()) / nIter
#         permLoss.backward(retain_graph=True)
#
#         totalLoss += permLoss.item()
#
#     return totalLoss

# def _loss(self, input, target):
#     # sum all paths losses * the path alphas multiplication
#     totalLoss = 0.0
#     nIter = min(self.nPerms, 1000)
#     for _ in range(nIter):
#         # for perm in product(*self.layersPerm):
#         perm = [randint(0, len(layer.alphas) - 1) for layer in self.layersList]
#         alphasProduct = 1.0
#         # set perm index in each layer
#         for i, p in enumerate(perm):
#             layer = self.layersList[i]
#             layer.curr_alpha_idx = p
#             probs = F.softmax(layer.alphas)
#             alphasProduct *= probs[p]
#
#         logits = self.forward(input)
#         # only the alphas are changing...
#         permLoss = (alphasProduct * self._criterion(logits, target, self.countBops()))
#         permLoss.backward(retain_graph=True)
#         totalLoss += permLoss.item()
#
#     # print('totalLoss:[{:.5f}]'.format(totalLoss))
#     return totalLoss

#
#     # logits = self.forward(input)
#     # return self._criterion(logits, target, self.countBops())

# def _loss(self, input, target):
#     # init how many samples per alpha
#     nSamples = self.nSamples
#     # init total loss
#     totalLoss = 0.0
#     # init loss samples list for ALL alphas
#     allLossSamples = []
#     for j, layer in enumerate(self.layersList):
#         # turn off coin toss for this layer
#         layer.alphas.requires_grad = False
#         # init layer alphas gradient
#         layerAlphasGrad = zeros(len(layer.alphas)).cuda()
#         # calc layer alphas softmax
#         probs = F.softmax(layer.alphas, dim=-1)
#
#         for i, alpha in enumerate(layer.alphas):
#             # select the specific alpha in this layer
#             layer.curr_alpha_idx = i
#             # init loss samples list
#             alphaLossSamples = []
#             for _ in range(nSamples):
#                 # forward through some path in model
#                 logits = self(input)
#                 # alphaLoss += self._criterion(logits, target, self.countBops()).detach()
#                 alphaLossSamples.append(self._criterion(logits, target, self.countBops()).detach())
#
#             # add current alpha loss samples to all loss samples list
#             allLossSamples.extend(alphaLossSamples)
#             # calc alpha average loss
#             alphaAvgLoss = sum(alphaLossSamples) / nSamples
#             layerAlphasGrad[i] = alphaAvgLoss
#             # add alpha loss to total loss
#             totalLoss += (alphaAvgLoss * probs[i])
#
#             # calc loss samples variance
#             lossVariance = [((x - alphaAvgLoss) ** 2) for x in alphaLossSamples]
#             lossVariance = sum(lossVariance) / (nSamples - 1)
#             # add alpha loss average to statistics
#             self.stats.containers[self.stats.alphaLossAvgKey][j][i].append(alphaAvgLoss.item())
#             # add alpha loss variance to statistics
#             self.stats.containers[self.stats.alphaLossVarianceKey][j][i].append(lossVariance.item())
#
#         # turn in coin toss for this layer
#         layer.alphas.requires_grad = True
#         # set layer alphas gradient
#         layer.alphas.grad = layerAlphasGrad
#
#         # add gradNorm to statistics
#         self.stats.containers[self.stats.gradNormKey][j].append(layerAlphasGrad.norm().item())
#
#     # average total loss
#     totalLoss /= self.nLayers()
#     # calc all loss samples average
#     nTotalSamples = len(allLossSamples)
#     allLossSamplesAvg = sum(allLossSamples) / nTotalSamples
#     # calc all loss samples variance
#     allLossSamples = [((x - allLossSamplesAvg) ** 2) for x in allLossSamples]
#     allLossSamplesVariance = (sum(allLossSamples) / (nTotalSamples - 1)).item()
#     # add all samples average & loss variance to statistics
#     self.stats.containers[self.stats.allSamplesLossAvgKey][0].append(allLossSamplesAvg)
#     self.stats.containers[self.stats.allSamplesLossVarianceKey][0].append(allLossSamplesVariance)
#
#     # subtract average total loss from every alpha gradient
#     for layer in self.layersList:
#         layer.alphas.grad -= totalLoss
#         # calc layer alphas softmax
#         probs = F.softmax(layer.alphas, dim=-1)
#         # multiply each grad by its probability
#         layer.alphas.grad *= probs
#
#     return totalLoss

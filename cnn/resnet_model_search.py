from torch import load as loadModel
from torch.nn import Module, Conv2d, AvgPool2d
import torch.nn.functional as F
from UNIQ.actquant import ActQuant
from UNIQ.quantize import backup_weights, restore_weights, quantize
from cnn.MixedOp import MixedConv, MixedConvWithReLU, MixedLinear


def save_quant_state(self, _):
    assert (self.noise is False)
    if self.quant and not self.noise and self.training:
        self.full_parameters = {}
        layers_list = self.get_layers_list()
        layers_steps = self.get_layers_steps(layers_list)
        assert (len(layers_steps) == 1)

        self.full_parameters = backup_weights(layers_steps[0], self.full_parameters)
        quantize(layers_steps[0], bitwidth=self.bitwidth[0])


def restore_quant_state(self, _, __):
    assert (self.noise is False)
    if self.quant and not self.noise and self.training:
        layers_list = self.get_layers_list()
        layers_steps = self.get_layers_steps(layers_list)
        assert (len(layers_steps) == 1)

        restore_weights(layers_steps[0], self.full_parameters)  # Restore the quantized layers


class BasicBlock(Module):
    def __init__(self, bitwidths, in_planes, out_planes, kernel_size, stride):
        super(BasicBlock, self).__init__()

        stride1 = stride if in_planes == out_planes else (stride + 1)

        self.block1 = MixedConvWithReLU(bitwidths, in_planes, out_planes, kernel_size, stride1, useResidual=False)
        self.block2 = MixedConvWithReLU(bitwidths, out_planes, out_planes, kernel_size, stride, useResidual=True)

        self.downsample = MixedConv(bitwidths, in_planes, out_planes, kernel_size, stride1) \
            if in_planes != out_planes else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.block1(x)
        out = self.block2(out, residual)

        return out


class ResNet(Module):
    nClasses = 10  # cifar-10

    def __init__(self, criterion, bitwidths):
        super(ResNet, self).__init__()

        # init MixedConvWithReLU layers list
        self.layersList = []

        self.block1 = MixedConvWithReLU(bitwidths, 3, 16, 3, 1)
        self.layersList.append(self.block1)

        layers = [
            BasicBlock(bitwidths, 16, 16, 3, 1),
            BasicBlock(bitwidths, 16, 16, 3, 1),
            BasicBlock(bitwidths, 16, 16, 3, 1),
            BasicBlock(bitwidths, 16, 32, 3, 1),
            BasicBlock(bitwidths, 32, 32, 3, 1),
            BasicBlock(bitwidths, 32, 32, 3, 1),
            BasicBlock(bitwidths, 32, 64, 3, 1),
            BasicBlock(bitwidths, 64, 64, 3, 1),
            BasicBlock(bitwidths, 64, 64, 3, 1)
        ]

        i = 2
        for l in layers:
            setattr(self, 'block{}'.format(i), l)
            i += 1

        self.avgpool = AvgPool2d(8)
        self.fc = MixedLinear(bitwidths, 64, self.nClasses)

        # build mixture layers list
        self.layersList = [m for m in self.modules() if isinstance(m, MixedOp)]

        # build alphas list, i.e. architecture parameters
        self._arch_parameters = [l.alphas for l in self.layersList]

        # set noise=True for 1st layer
        for op in self.layersList[0].ops:
            if op.quant:
                op.noise = True

        # init criterion
        self._criterion = criterion

        # set learnable parameters
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]
        # update model parameters() function
        self.parameters = self.getLearnableParams

        # init number of layers we have completed its quantization
        self.nLayersQuantCompleted = 0

    def nLayers(self):
        return len(self.layersList)

    def forward(self, x):
        out = self.block1(x)

        blockNum = 2
        b = getattr(self, 'block{}'.format(blockNum))
        while b is not None:
            out = b(out)

            # move to next block
            blockNum += 1
            b = getattr(self, 'block{}'.format(blockNum), None)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def arch_parameters(self):
        return self._arch_parameters

    def getLearnableParams(self):
        return self.learnable_params

    # return top k operations per layer
    def topOps(self, k):
        top = []
        for layer in self.layersList:
            # calc weights from alphas and sort them
            weights = F.softmax(layer.alphas, dim=-1)
            # weights = layer.alphas
            wSorted, wIndices = weights.sort(descending=True)
            # keep only top-k
            wSorted = wSorted[:k]
            wIndices = wIndices[:k]
            # add to top
            top.append([(w.item(), layer.alphas[i], layer.ops[i]) for w, i in zip(wSorted, wIndices)])

        return top

    def switch_stage(self, logger=None):
        # TODO: freeze stage alphas as well ???
        if self.nLayersQuantCompleted + 1 < len(self.layersList):
            layer = self.layersList[self.nLayersQuantCompleted]
            for op in layer.ops:
                # turn off noise in op
                assert (op.noise is True)
                op.noise = False

                # set pre & post quantization hooks, from now on we want to quantize these ops
                op.register_forward_pre_hook(save_quant_state)
                op.register_forward_hook(restore_quant_state)

                # turn off gradients
                for m in op.modules():
                    if isinstance(m, Conv2d):
                        for param in m.parameters():
                            param.requires_grad = False
                    elif isinstance(m, ActQuant):
                        m.quatize_during_training = True
                        m.noise_during_training = False

            # update learnable parameters
            self.learnable_params = [param for param in self.parameters() if param.requires_grad]

            # we have completed quantization of one more layer
            self.nLayersQuantCompleted += 1

            # turn on noise in the new layer we want to quantize
            layer = self.layersList[self.nLayersQuantCompleted]
            for op in layer.ops:
                op.noise = True

            if logger:
                logger.info('Switching stage, nLayersQuantCompleted:[{}], learnable_params:[{}]'
                            .format(self.nLayersQuantCompleted, len(self.learnable_params)))

    def loadFromCheckpoint(self, path, logger, gpu):
        checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
        # split state_dict keys by model layers
        layerKeys = {}  # collect ALL layer keys in state_dict
        layerOp0Keys = {}  # collect ONLY layer.ops.0. keys in state_dict, for duplication to the rest of layer ops
        token = '.ops.'
        for key in checkpoint['state_dict'].keys():
            prefix = key[:key.index(token)]
            isKey0 = prefix + token + '0.' in key
            if prefix in layerKeys:
                layerKeys[prefix].append(key)
                if isKey0: layerOp0Keys[prefix].append(key)
            else:
                layerKeys[prefix] = [key]
                if isKey0: layerOp0Keys[prefix] = [key]
        # duplicate state_dict values according to number of ops in each model layer
        for layerKey in layerKeys.keys():
            # init path to layer
            layerPath = [p for p in layerKey.split('.')]
            # get layer by walking through path
            layer = self
            for p in layerPath:
                layer = getattr(layer, p)
            # add missing layer operations to state_dict
            for stateKey in layerOp0Keys[layerKey]:
                for i in range(len(layer.ops)):
                    newKey = stateKey.replace(layerKey + token + '0.', layerKey + token + '{}.'.format(i))
                    if newKey not in layerKeys[layerKey]:
                        checkpoint['state_dict'][newKey] = checkpoint['state_dict'][stateKey]

        # load model weights
        self.load_state_dict(checkpoint['state_dict'])
        # load model alphas
        # if 'alphas' in checkpoint:
        #     for i, l in enumerate(self.layersList):
        #         layerChkpntAlphas = checkpoint['alphas'][i]
        #         assert (layerChkpntAlphas.size() <= l.alphas.size())
        #         l.alphas = layerChkpntAlphas.expand_as(l.alphas)

        # load nLayersQuantCompleted
        # if 'nLayersQuantCompleted' in checkpoint:
        #     self.nLayersQuantCompleted = checkpoint['nLayersQuantCompleted']

        logger.info('Loaded model from [{}]'.format(path))
        logger.info('checkpoint validation accuracy:[{:.5f}]'.format(checkpoint['best_prec1']))

# def loadFromCheckpoint(self, path, logger, gpu):
#     checkpoint = loadModel(path, map_location=lambda storage, loc: storage.cuda(gpu))
#
#     for i in range(self.block1.numOfOps()):
#         self.block1.ops[i]._modules['op'][0][0].weight.data.copy_(checkpoint['state_dict']['conv1.weight'])
#         self.block1.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['bn1.weight'])
#         self.block1.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['bn1.bias'])
#         self.block1.ops[i]._modules['op'][0][1].running_var.data.copy_(checkpoint['state_dict']['bn1.running_var'])
#         self.block1.ops[i]._modules['op'][0][1].running_mean.data.copy_(checkpoint['state_dict']['bn1.running_mean'])
#
#     for i in range(self.block2.block1.numOfOps()):
#         self.block2.block1.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer1.0.conv1.weight'])
#         self.block2.block1.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer1.0.bn1.weight'])
#         self.block2.block1.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer1.0.bn1.bias'])
#         self.block2.block1.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer1.0.bn1.running_var'])
#         self.block2.block1.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer1.0.bn1.running_mean'])
#
#     for i in range(self.block2.block2.numOfOps()):
#         self.block2.block2.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer1.0.conv2.weight'])
#         self.block2.block2.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer1.0.bn2.weight'])
#         self.block2.block2.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer1.0.bn2.bias'])
#         self.block2.block2.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer1.0.bn2.running_var'])
#         self.block2.block2.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer1.0.bn2.running_mean'])
#
#     for i in range(self.block3.block1.numOfOps()):
#         self.block3.block1.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer1.1.conv1.weight'])
#         self.block3.block1.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer1.1.bn1.weight'])
#         self.block3.block1.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer1.1.bn1.bias'])
#         self.block3.block1.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer1.1.bn1.running_var'])
#         self.block3.block1.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer1.1.bn1.running_mean'])
#
#     for i in range(self.block3.block2.numOfOps()):
#         self.block3.block2.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer1.1.conv2.weight'])
#         self.block3.block2.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer1.1.bn2.weight'])
#         self.block3.block2.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer1.1.bn2.bias'])
#         self.block3.block2.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer1.1.bn2.running_var'])
#         self.block3.block2.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer1.1.bn2.running_mean'])
#
#     for i in range(self.block4.block1.numOfOps()):
#         self.block4.block1.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer1.2.conv1.weight'])
#         self.block4.block1.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer1.2.bn1.weight'])
#         self.block4.block1.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer1.2.bn1.bias'])
#         self.block4.block1.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer1.2.bn1.running_var'])
#         self.block4.block1.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer1.2.bn1.running_mean'])
#
#     for i in range(self.block4.block2.numOfOps()):
#         self.block4.block2.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer1.2.conv2.weight'])
#         self.block4.block2.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer1.2.bn2.weight'])
#         self.block4.block2.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer1.2.bn2.bias'])
#         self.block4.block2.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer1.2.bn2.running_var'])
#         self.block4.block2.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer1.2.bn2.running_mean'])
#
#     for i in range(self.block5.block1.numOfOps()):
#         self.block5.block1.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer2.0.conv1.weight'])
#         self.block5.block1.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer2.0.bn1.weight'])
#         self.block5.block1.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer2.0.bn1.bias'])
#         self.block5.block1.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer2.0.bn1.running_var'])
#         self.block5.block1.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer2.0.bn1.running_mean'])
#
#     for i in range(self.block5.block2.numOfOps()):
#         self.block5.block2.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer2.0.conv2.weight'])
#         self.block5.block2.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer2.0.bn2.weight'])
#         self.block5.block2.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer2.0.bn2.bias'])
#         self.block5.block2.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer2.0.bn2.running_var'])
#         self.block5.block2.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer2.0.bn2.running_mean'])
#
#     for i in range(self.block5.downsample.numOfOps()):
#         self.block5.downsample.ops[i]._modules['op'][0].weight.data.copy_(
#             checkpoint['state_dict']['layer2.0.downsample.0.weight'])
#         self.block5.downsample.ops[i]._modules['op'][1].weight.data.copy_(
#             checkpoint['state_dict']['layer2.0.downsample.1.weight'])
#         self.block5.downsample.ops[i]._modules['op'][1].bias.data.copy_(
#             checkpoint['state_dict']['layer2.0.downsample.1.bias'])
#         self.block5.downsample.ops[i]._modules['op'][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer2.0.downsample.1.running_var'])
#         self.block5.downsample.ops[i]._modules['op'][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer2.0.downsample.1.running_mean'])
#
#     for i in range(self.block6.block1.numOfOps()):
#         self.block6.block1.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer2.1.conv1.weight'])
#         self.block6.block1.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer2.1.bn1.weight'])
#         self.block6.block1.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer2.1.bn1.bias'])
#         self.block6.block1.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer2.1.bn1.running_var'])
#         self.block6.block1.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer2.1.bn1.running_mean'])
#
#     for i in range(self.block6.block2.numOfOps()):
#         self.block6.block2.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer2.1.conv2.weight'])
#         self.block6.block2.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer2.1.bn2.weight'])
#         self.block6.block2.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer2.1.bn2.bias'])
#         self.block6.block2.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer2.1.bn2.running_var'])
#         self.block6.block2.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer2.1.bn2.running_mean'])
#
#     for i in range(self.block7.block1.numOfOps()):
#         self.block7.block1.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer2.2.conv1.weight'])
#         self.block7.block1.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer2.2.bn1.weight'])
#         self.block7.block1.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer2.2.bn1.bias'])
#         self.block7.block1.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer2.2.bn1.running_var'])
#         self.block7.block1.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer2.2.bn1.running_mean'])
#
#     for i in range(self.block7.block2.numOfOps()):
#         self.block7.block2.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer2.2.conv2.weight'])
#         self.block7.block2.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer2.2.bn2.weight'])
#         self.block7.block2.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer2.2.bn2.bias'])
#         self.block7.block2.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer2.2.bn2.running_var'])
#         self.block7.block2.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer2.2.bn2.running_mean'])
#
#     for i in range(self.block8.block1.numOfOps()):
#         self.block8.block1.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer3.0.conv1.weight'])
#         self.block8.block1.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer3.0.bn1.weight'])
#         self.block8.block1.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer3.0.bn1.bias'])
#         self.block8.block1.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer3.0.bn1.running_var'])
#         self.block8.block1.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer3.0.bn1.running_mean'])
#
#     for i in range(self.block8.block2.numOfOps()):
#         self.block8.block2.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer3.0.conv2.weight'])
#         self.block8.block2.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer3.0.bn2.weight'])
#         self.block8.block2.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer3.0.bn2.bias'])
#         self.block8.block2.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer3.0.bn2.running_var'])
#         self.block8.block2.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer3.0.bn2.running_mean'])
#
#     for i in range(self.block8.downsample.numOfOps()):
#         self.block8.downsample.ops[i]._modules['op'][0].weight.data.copy_(
#             checkpoint['state_dict']['layer3.0.downsample.0.weight'])
#         self.block8.downsample.ops[i]._modules['op'][1].weight.data.copy_(
#             checkpoint['state_dict']['layer3.0.downsample.1.weight'])
#         self.block8.downsample.ops[i]._modules['op'][1].bias.data.copy_(
#             checkpoint['state_dict']['layer3.0.downsample.1.bias'])
#         self.block8.downsample.ops[i]._modules['op'][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer3.0.downsample.1.running_var'])
#         self.block8.downsample.ops[i]._modules['op'][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer3.0.downsample.1.running_mean'])
#
#     for i in range(self.block9.block1.numOfOps()):
#         self.block9.block1.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer3.1.conv1.weight'])
#         self.block9.block1.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer3.1.bn1.weight'])
#         self.block9.block1.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer3.1.bn1.bias'])
#         self.block9.block1.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer3.1.bn1.running_var'])
#         self.block9.block1.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer3.1.bn1.running_mean'])
#
#     for i in range(self.block9.block2.numOfOps()):
#         self.block9.block2.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer3.1.conv2.weight'])
#         self.block9.block2.ops[i]._modules['op'][0][1].weight.data.copy_(checkpoint['state_dict']['layer3.1.bn2.weight'])
#         self.block9.block2.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer3.1.bn2.bias'])
#         self.block9.block2.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer3.1.bn2.running_var'])
#         self.block9.block2.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer3.1.bn2.running_mean'])
#
#     for i in range(self.block10.block1.numOfOps()):
#         self.block10.block1.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer3.2.conv1.weight'])
#         self.block10.block1.ops[i]._modules['op'][0][1].weight.data.copy_(
#             checkpoint['state_dict']['layer3.2.bn1.weight'])
#         self.block10.block1.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer3.2.bn1.bias'])
#         self.block10.block1.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer3.2.bn1.running_var'])
#         self.block10.block1.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer3.2.bn1.running_mean'])
#
#     for i in range(self.block10.block2.numOfOps()):
#         self.block10.block2.ops[i]._modules['op'][0][0].weight.data.copy_(
#             checkpoint['state_dict']['layer3.2.conv2.weight'])
#         self.block10.block2.ops[i]._modules['op'][0][1].weight.data.copy_(
#             checkpoint['state_dict']['layer3.2.bn2.weight'])
#         self.block10.block2.ops[i]._modules['op'][0][1].bias.data.copy_(checkpoint['state_dict']['layer3.2.bn2.bias'])
#         self.block10.block2.ops[i]._modules['op'][0][1].running_var.data.copy_(
#             checkpoint['state_dict']['layer3.2.bn2.running_var'])
#         self.block10.block2.ops[i]._modules['op'][0][1].running_mean.data.copy_(
#             checkpoint['state_dict']['layer3.2.bn2.running_mean'])
#
#     for i in range(self.fc.numOfOps()):
#         self.fc.ops[i]._modules['op'].weight.data.copy_(checkpoint['state_dict']['fc.weight'])
#         self.fc.ops[i]._modules['op'].bias.data.copy_(checkpoint['state_dict']['fc.bias'])
#
#     logger.info('Loaded model from [{}]'.format(path))

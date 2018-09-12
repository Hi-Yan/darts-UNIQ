from collections import OrderedDict

from torch.nn import Sequential, Conv2d

from cnn.MixedOp import MixedConvWithReLU, MixedLinear
from cnn.models.BaseNet import BaseNet, save_quant_state, restore_quant_state, ActQuant


class TinyNet(BaseNet):
    def __init__(self, args):
        super(TinyNet, self).__init__(args, initLayersParams=(args.bitwidth, args.kernel))

        for layer in self.layersList:
            # turn on noise
            for op in layer.ops:
                op.noise = op.quant

    def initLayers(self, params):
        bitwidths, kernel_sizes = params

        # self.features = nn.Sequential(
        #     MixedConv(bitwidths, 3, 16, kernel_sizes, stride=2), nn.ReLU(inplace=True),
        #     MixedConv(bitwidths, 16, 32, kernel_sizes, stride=2), nn.ReLU(inplace=True),
        #     MixedConv(bitwidths, 32, 64, kernel_sizes, stride=2), nn.ReLU(inplace=True),
        #     MixedConv(bitwidths, 64, 128, kernel_sizes, stride=2), nn.ReLU(inplace=True)
        # )
        self.features = Sequential(
            MixedConvWithReLU(bitwidths, 3, 16, kernel_sizes, stride=2),
            MixedConvWithReLU(bitwidths, 16, 32, kernel_sizes, stride=2),
            MixedConvWithReLU(bitwidths, 32, 64, kernel_sizes, stride=2),
            MixedConvWithReLU(bitwidths, 64, 128, kernel_sizes, stride=2),
        )
        self.fc = MixedLinear(bitwidths, 512, 10)

        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        #     nn.Conv2d(16, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        # )
        # self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

    def switch_stage(self, logger=None):
        pass

    def turnOnAlphas(self):
        self.learnable_alphas = []
        for layer in self.layersList:
            # turn on alphas gradients
            layer.alphas.requires_grad = True
            self.learnable_alphas.append(layer.alphas)

            for op in layer.ops:
                # turn off noise in op
                assert (op.noise is True)
                op.noise = False

                # set pre & post quantization hooks, from now on we want to quantize these ops
                op.hookHandlers.append(op.register_forward_pre_hook(save_quant_state))
                op.hookHandlers.append(op.register_forward_hook(restore_quant_state))

                # turn off weights gradients
                for m in op.modules():
                    if isinstance(m, Conv2d):
                        for param in m.parameters():
                            param.requires_grad = False
                    elif isinstance(m, ActQuant):
                        m.quatize_during_training = True
                        m.noise_during_training = False

        # update learnable parameters
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]

    def turnOnWeights(self):
        for layer in self.layersList:
            for op in layer.ops:
                assert (op.noise is False)
                # turn on operations noise
                op.noise = True
                # remove hooks
                for handler in op.hookHandlers:
                    handler.remove()
                # clear hooks handlers list
                op.hookHandlers.clear()
                # turn on operations gradients
                for m in op.modules():
                    if isinstance(m, Conv2d):
                        for param in m.parameters():
                            param.requires_grad = True
                    elif isinstance(m, ActQuant):
                        m.quatize_during_training = False
                        m.noise_during_training = True

        # update learnable parameters
        self.learnable_params = [param for param in self.parameters() if param.requires_grad]

    # load original pre_trained model of UNIQ
    def loadUNIQPre_trained(self, chckpntDict):
        map = {}
        map['features.0'] = 'features.0.ops.0.op.0.0'
        map['features.1'] = 'features.0.ops.0.op.0.1'

        map['features.3'] = 'features.1.ops.0.op.0.0'
        map['features.4'] = 'features.1.ops.0.op.0.1'

        map['features.6'] = 'features.2.ops.0.op.0.0'
        map['features.7'] = 'features.2.ops.0.op.0.1'

        map['features.9'] = 'features.3.ops.0.op.0.0'
        map['features.10'] = 'features.3.ops.0.op.0.1'

        map['fc'] = 'fc.ops.0.op'

        newStateDict = OrderedDict()

        # d = self.state_dict()

        token = '.ops.'
        for key in chckpntDict.keys():
            if 'num_batches_tracked' in key:
                continue

            prefix = key[:key.rindex('.')]
            suffix = key[key.rindex('.'):]
            newKey = map[prefix]
            # find new key layer
            newKeyOp = newKey[:newKey.index(token)]
            # init path to layer
            layerPath = [p for p in newKeyOp.split('.')]
            # get layer by walking through path
            layer = self
            for p in layerPath:
                layer = getattr(layer, p)
            # update layer ops
            for i in range(len(layer.ops)):
                newStateDict[newKey + suffix] = chckpntDict[key]
                newKey = newKey.replace(newKeyOp + token + '{}.'.format(i), newKeyOp + token + '{}.'.format(i + 1))

        # load model weights
        self.load_state_dict(newStateDict)

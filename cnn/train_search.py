import os
from sys import exit
from time import time, strftime
import glob
import numpy as np
import argparse

from torch.nn import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets.cifar import CIFAR10
import torch.backends.cudnn as cudnn
from torch.cuda import is_available, set_device
from torch.cuda import manual_seed as cuda_manual_seed
from torch import manual_seed as torch_manual_seed
from torch import no_grad
from torch.optim import SGD, Adam
from torch.autograd.variable import Variable

from cnn.utils import create_exp_dir, count_parameters_in_MB, _data_transforms_cifar10, accuracy, AvgrageMeter, save
from cnn.utils import initLogger, printModelToFile, initTrainLogger, logDominantQuantizedOp, save_checkpoint
from cnn.model_search import Network
from cnn.resnet_model_search import ResNet
from cnn.architect import Architect


def parseArgs():
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--data', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1E-8, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id, e.g. 0,1,3')
    parser.add_argument('--epochs', type=str, default='5',
                        help='num of training epochs per layer, as list, e.g. 5,4,3,8,6.'
                             'If len(epochs)<len(layers) then last value is used for rest of the layers')
    parser.add_argument('--workers', type=int, default=1, choices=range(1, 32), help='num of workers')
    # parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    # parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    # parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--propagate', action='store_true', default=False, help='print to stdout')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    parser.add_argument('--checkpoint', type=str,
                        default='/home/yochaiz/darts/cnn/pre_trained_models/resnet_18/model_opt.pth.tar')
    parser.add_argument('--nBitsMin', type=int, default=1, choices=range(1, 32 + 1), help='min number of bits')
    parser.add_argument('--nBitsMax', type=int, default=3, choices=range(1, 32 + 1), help='max number of bits')
    args = parser.parse_args()

    # convert epochs to list
    args.epochs = [int(i) for i in args.epochs.split(',')]

    # update GPUs list
    if type(args.gpu) is str:
        args.gpu = [int(i) for i in args.gpu.split(',')]

    args.device = 'cuda:' + str(args.gpu[0])

    args.save = 'results/search-{}-{}'.format(args.save, strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    return args


def train(train_queue, search_queue, args, model, architect, criterion, optimizer, lr, logger):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    grad = AvgrageMeter()

    model.train()
    nBatches = len(train_queue)

    for step, (input, target) in enumerate(train_queue):
        startTime = time()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(search_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(async=True)

        arch_grad_norm = architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        grad.update(arch_grad_norm)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        endTime = time()

        if step % args.report_freq == 0:
            logger.info('train [{}/{}] Loss:[{:.5f}] Accuracy:[{:.3f}] time:[{:.5f}]'
                        .format(step, nBatches, objs.avg, top1.avg, endTime - startTime))

    return top1.avg, objs.avg


def infer(valid_queue, args, model, criterion, logger):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    model.eval()
    nBatches = len(valid_queue)

    with no_grad():
        for step, (input, target) in enumerate(valid_queue):
            startTime = time()

            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda(async=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            endTime = time()

            if step % args.report_freq == 0:
                logger.info('validation [{}/{}] Loss:[{:.5f}] Accuracy:[{:.3f}] time:[{:.5f}]'.
                            format(step, nBatches, objs.avg, top1.avg, endTime - startTime))

    return top1.avg, objs.avg


args = parseArgs()
print(args)
logger = initLogger(args.save, args.propagate)
CIFAR_CLASSES = 10

if not is_available():
    logger.info('no gpu device available')
    exit(1)

np.random.seed(args.seed)
set_device(args.gpu[0])
cudnn.benchmark = True
torch_manual_seed(args.seed)
cudnn.enabled = True
cuda_manual_seed(args.seed)

criterion = CrossEntropyLoss()
criterion = criterion.cuda()
# criterion = criterion.to(args.device)
# model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
model = ResNet(criterion, args.nBitsMin, args.nBitsMax)
# model = DataParallel(model, args.gpu)
model = model.cuda()
# model = model.to(args.device)

# load full-precision model
path = '/home/yochaiz/darts/cnn/results/search-EXP-20180729-175054/model_opt.pth.tar'
model.loadFromCheckpoint(path, logger, args.gpu[0])

# print some attributes
printModelToFile(model, args.save)
logger.info('GPU:{}'.format(args.gpu))
logger.info("args = %s", args)
logger.info("param size = %fMB", count_parameters_in_MB(model))
logger.info('Learnable params:[{}]'.format(len(model.learnable_params)))
logger.info('alphas tensor size:[{}]'.format(model.arch_parameters()[0].size()))

optimizer = SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = Adam(model.parameters(), lr=args.learning_rate,
#                  betas=(0.5, 0.999), weight_decay=args.weight_decay)

train_transform, valid_transform = _data_transforms_cifar10(args)
train_data = CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
valid_data = CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

#### narrow data for debug purposes
# train_data.train_data = train_data.train_data[0:640]
# train_data.train_labels = train_data.train_labels[0:640]
# valid_data.test_data = valid_data.test_data[0:320]
# valid_data.test_labels = valid_data.test_labels[0:320]
####

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))

train_queue = DataLoader(train_data, batch_size=args.batch_size,
                         sampler=SubsetRandomSampler(indices[:split]), pin_memory=True, num_workers=args.workers)

search_queue = DataLoader(train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices[split:num_train]),
                          pin_memory=True, num_workers=args.workers)

valid_queue = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                         pin_memory=True, num_workers=args.workers)

# extend epochs list as number of model layers
while len(args.epochs) < model.nLayers():
    args.epochs.append(args.epochs[-1])
# init epochs number where we have to switch stage in
epochsSwitchStage = [0]
for e in args.epochs:
    epochsSwitchStage.append(e + epochsSwitchStage[-1])
# total number of epochs is the last value in epochsSwitchStage
nEpochs = epochsSwitchStage[-1] + 1
# remove epoch 0 from list, we don't want to switch stage at the beginning
epochsSwitchStage = epochsSwitchStage[1:]

logger.info('nEpochs:[{}]'.format(nEpochs))
logger.info('epochsSwitchStage:{}'.format(epochsSwitchStage))

scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)
architect = Architect(model, args)

best_prec1 = 0.0

for epoch in range(1, nEpochs + 1):
    trainLogger = initTrainLogger(str(epoch), args.save, args.propagate)

    scheduler.step()
    lr = scheduler.get_lr()[0]

    trainLogger.info('optimizer_lr:[{:.5f}], scheduler_lr:[{:.5f}]'.format(optimizer.defaults['lr'], lr))

    # print(F.softmax(model.alphas_normal, dim=-1))
    # print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_loss = train(train_queue, search_queue, args, model, architect, criterion, optimizer, lr, trainLogger)

    # log accuracy, loss, etc.
    message = 'Epoch:[{}] , training accuracy:[{:.3f}] , training loss:[{:.3f}] , optimizer_lr:[{:.5f}], scheduler_lr:[{:.5f}]' \
        .format(epoch, train_acc, train_loss, optimizer.defaults['lr'], lr)
    logger.info(message)
    trainLogger.info(message)

    # log dominant QuantizedOp in each layer
    logDominantQuantizedOp(model, k=2, logger=trainLogger)

    # save model checkpoint
    save_checkpoint(args.save, model, epoch, is_best=False)

    # switch stage, i.e. freeze one more layer
    if epoch in epochsSwitchStage:
        # validation
        valid_acc, valid_loss = infer(valid_queue, args, model, criterion, trainLogger)
        message = 'Epoch:[{}] , validation accuracy:[{:.3f}] , validation loss:[{:.3f}]'.format(epoch, valid_acc,
                                                                                                valid_loss)
        logger.info(message)
        trainLogger.info(message)

        # save model checkpoint
        is_best = valid_acc > best_prec1
        best_prec1 = max(valid_acc, best_prec1)
        save_checkpoint(args.save, model, epoch, is_best)

        # switch stage
        model.switch_stage(trainLogger)
        # update optimizer & scheduler due to update in learnable params
        optimizer = SGD(model.parameters(), scheduler.get_lr()[0],
                        momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)

save(model, os.path.join(args.save, 'weights.pt'))

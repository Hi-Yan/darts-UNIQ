from time import time
from abc import abstractmethod
from os import makedirs, path

from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import no_grad, tensor, int32
from torch.optim import SGD
from torch.autograd.variable import Variable
from torch.nn import CrossEntropyLoss

from cnn.utils import accuracy, AvgrageMeter, load_data,  logDominantQuantizedOp, save_checkpoint
from cnn.utils import sendDataEmail, logForwardCounters, models, logParameters
from cnn.HtmlLogger import HtmlLogger


class TrainRegime:
    trainLossKey = 'Training loss'
    trainAccKey = 'Training acc'
    validLossKey = 'Validation loss'
    validAccKey = 'Validation acc'
    archLossKey = 'Arch loss'
    crossEntropyKey = 'CrossEntropy loss'
    bopsLossKey = 'Bops loss'
    epochNumKey = 'Epoch #'
    batchNumKey = 'Batch #'
    pathBopsRatioKey = 'Path bops ratio'
    optBopsRatioKey = 'Optimal bops ratio'
    validBopsRatioKey = 'Validation bops ratio'
    timeKey = 'Time'
    lrKey = 'Optimizer lr'
    bitwidthKey = 'Bitwidth'
    statsKey = 'Stats'
    forwardCountersKey = 'Forward counters'

    # init formats for keys
    formats = {validLossKey: '{:.5f}', validAccKey: '{:.3f}', optBopsRatioKey: '{:.3f}', timeKey: '{:.3f}', archLossKey: '{:.5f}', lrKey: '{:.5f}',
               trainLossKey: '{:.5f}', trainAccKey: '{:.3f}', pathBopsRatioKey: '{:.3f}', validBopsRatioKey: '{:.3f}', crossEntropyKey: '{:.5f}',
               bopsLossKey: '{:.5f}'}

    initWeightsTrainTableTitle = 'Initial weights training'
    k = 2
    alphasTableTitle = 'Alphas (top [{}])'.format(k)

    colsTrainWeights = [batchNumKey, trainLossKey, trainAccKey, bitwidthKey, pathBopsRatioKey, statsKey, timeKey]
    colsMainInitWeightsTrain = [epochNumKey, trainLossKey, trainAccKey, validLossKey, validAccKey, validBopsRatioKey, lrKey]
    colsTrainAlphas = [batchNumKey, archLossKey, crossEntropyKey, bopsLossKey, alphasTableTitle, pathBopsRatioKey, forwardCountersKey, timeKey]
    colsValidation = [batchNumKey, validLossKey, validAccKey, statsKey, timeKey]
    colsValidationStatistics = [forwardCountersKey, bitwidthKey, validBopsRatioKey]
    colsMainLogger = [epochNumKey, archLossKey, trainLossKey, trainAccKey, validLossKey, validAccKey, validBopsRatioKey, bitwidthKey, lrKey]

    def __init__(self, args, logger):
        # build model for uniform distribution of bits
        modelClass = models.__dict__[args.model]
        # init model
        model = modelClass(args)
        model = model.cuda()
        # init baseline bops
        baselineBops = model.calcBaselineBops()
        args.baselineBops = baselineBops[args.baselineBits[0]]
        # plot baselines bops
        model.stats.addBaselineBopsData(args, baselineBops)
        # load partition if exists
        if args.partition is not None:
            assert (isinstance(args.partition, list))
            # convert partition to tensors
            for i, p in enumerate(args.partition):
                args.partition[i] = tensor(p, dtype=int32).cuda()
            # set filters by partition
            model.setFiltersByPartition(args.partition, loggerFuncs=[lambda msg: logger.addInfoTable('Partition', [[msg]])])

        # # ========================== DEBUG ===============================
        # self.model = model
        # self.args = args
        # self.logger = logger
        # return
        # ==================================================================
        # load data
        self.train_queue, self.search_queue, self.valid_queue, self.statistics_queue = load_data(args)
        # load pre-trained full-precision model
        args.loadedOpsWithDiffWeights = model.loadPreTrained(args.pre_trained, logger, args.gpu[0])

        # log parameters
        logParameters(logger, args, model)

        self.args = args
        self.model = model
        self.modelClass = modelClass
        self.logger = logger

        # init email time
        self.lastMailTime = time()
        self.secondsBetweenMails = 1 * 3600

        # number of batches for allocation as optimal model in order to train it from full-precision
        self.nBatchesOptModel = 20

        # init train optimal model counter.
        # each key is an allocation, and the map hold a counter per key, how many batches this allocation is optimal
        self.optModelBitwidthCounter = {}
        # init optimal model training queue, in case we send too many jobs to server
        self.optModelTrainingQueue = []

        self.trainFolderPath = '{}/{}'.format(args.save, args.trainFolder)

        # init cross entropy loss
        self.cross_entropy = CrossEntropyLoss().cuda()

        # init checkpoints dictionary
        self.optimalModelCheckpoint = (None, None)

        # extend epochs list as number of model layers
        while len(args.epochs) < model.nLayers():
            args.epochs.append(args.epochs[-1])
        # init epochs number where we have to switch stage in
        epochsSwitchStage = [0]
        for e in args.epochs:
            epochsSwitchStage.append(e + epochsSwitchStage[-1])
        # on epochs we learn only Linear layer, infer in every epoch
        # for _ in range(args.epochs[-1]):
        for _ in range(10):
            epochsSwitchStage.append(epochsSwitchStage[-1] + 1)

        # total number of epochs is the last value in epochsSwitchStage
        nEpochs = epochsSwitchStage[-1]
        # remove epoch 0 from list, we don't want to switch stage at the beginning
        epochsSwitchStage = epochsSwitchStage[1:]

        logger.addInfoTable('Epochs', [['nEpochs', '{}'.format(nEpochs)], ['epochsSwitchStage', '{}'.format(epochsSwitchStage)]])

        # init epoch
        self.epoch = 0
        self.nEpochs = nEpochs
        self.epochsSwitchStage = epochsSwitchStage

        # if we loaded ops in the same layer with the same weights, then we loaded the optimal full precision model,
        # therefore we have to train the weights for each QuantizedOp
        if (args.loadedOpsWithDiffWeights is False) and args.init_weights_train:
            self.epoch = self.initialWeightsTraining(trainFolderName='init_weights_train')
        else:
            rows = [['Switching stage']]
            # we loaded ops in the same layer with different weights, therefore we just have to switch_stage
            switchStageFlag = True
            while switchStageFlag:
                switchStageFlag = model.switch_stage([lambda msg: rows.append([msg])])
            # create info table
            logger.addInfoTable(self.initWeightsTrainTableTitle, rows)

        # init logger data table
        logger.createDataTable('Summary', self.colsMainLogger)

    @abstractmethod
    def train(self):
        raise NotImplementedError('subclasses must override train()!')

    def initialWeightsTraining(self, trainFolderName, filename=None):
        model = self.model
        args = self.args
        nEpochs = self.nEpochs
        logger = self.logger

        # create train folder
        folderPath = '{}/{}'.format(self.trainFolderPath, trainFolderName)
        if not path.exists(folderPath):
            makedirs(folderPath)

        # init optimizer
        optimizer = SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        # init scheduler
        scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)

        # init validation best precision value
        best_prec1 = 0.0
        best_valid_loss = 0.0
        is_best = False

        epoch = 0
        # init table in main logger
        logger.createDataTable(self.initWeightsTrainTableTitle, self.colsMainInitWeightsTrain)

        for epoch in range(1, nEpochs + 1):
            scheduler.step()
            lr = scheduler.get_lr()[0]

            trainLogger = HtmlLogger(folderPath, str(epoch))
            trainLogger.addInfoTable('Learning rates', [
                ['optimizer_lr', self.formats[self.lrKey].format(optimizer.param_groups[0]['lr'])],
                ['scheduler_lr', self.formats[self.lrKey].format(lr)]
            ])

            # set loggers dictionary
            loggersDict = dict(train=trainLogger)
            # training
            print('========== Epoch:[{}] =============='.format(epoch))
            trainData = self.trainWeights(optimizer, epoch, loggersDict)

            # add epoch number
            trainData[self.epochNumKey] = epoch
            # add learning rate
            trainData[self.lrKey] = self.formats[self.lrKey].format(optimizer.param_groups[0]['lr'])

            # switch stage, i.e. freeze one more layer
            if (epoch in self.epochsSwitchStage) or (epoch == nEpochs):
                # validation
                valid_acc, valid_loss, validData = self.infer(model.setFiltersByAlphas, epoch, loggersDict)

                # merge trainData with validData
                for k, v in validData.items():
                    trainData[k] = v

                # switch stage
                switchStageFlag = model.switch_stage(loggerFuncs=[lambda msg: trainLogger.addInfoTable(title='Switching stage', rows=[[msg]])])
                # update optimizer only if we changed model learnable params
                if switchStageFlag:
                    # update optimizer & scheduler due to update in learnable params
                    optimizer = SGD(model.parameters(), scheduler.get_lr()[0], momentum=args.momentum, weight_decay=args.weight_decay)
                    scheduler = CosineAnnealingLR(optimizer, float(nEpochs), eta_min=args.learning_rate_min)
                    scheduler.step()
                else:
                    # update best precision only after switching stage is complete
                    is_best = valid_acc > best_prec1
                    if is_best:
                        best_prec1 = valid_acc
                        best_valid_loss = valid_loss
                # save model checkpoint
                checkpoint, (_, optimalPath) = save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best, filename)
                if is_best:
                    assert (optimalPath is not None)
                    self.optimalModelCheckpoint = checkpoint, optimalPath
            else:
                # save model checkpoint
                save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best=False, filename=filename)

            # add data to main logger table
            logger.addDataRow(trainData)

        # add optimal accuracy
        summaryRow = {self.epochNumKey: 'Optimal', self.validAccKey: best_prec1, self.validLossKey: best_valid_loss}
        self._applyFormats(summaryRow)
        logger.addSummaryDataRow(summaryRow)

        # # save pre-trained checkpoint
        # save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best=False, filename='pre_trained')

        # save optimal validation values
        setattr(args, self.validAccKey, best_prec1)
        setattr(args, self.validLossKey, best_valid_loss)

        return epoch

    def sendEmail(self, nEpoch, batchNum, nBatches):
        body = ['Hi', 'Files are attached.', 'Epoch:[{}]  Batch:[{}/{}]'.format(nEpoch, batchNum, nBatches)]
        content = ''
        for line in body:
            content += line + '\n'

        sendDataEmail(self.model, self.args, self.logger, content)

    # apply defined formats on dict values by keys
    def _applyFormats(self, dict):
        for k in dict.keys():
            if k in self.formats:
                dict[k] = self.formats[k].format(dict[k])

    @staticmethod
    def addModelUNIQstatusTable(model, logger, title):
        # init UNIQ params in MixedLayer
        params = ['quantized', 'added_noise']
        # collect UNIQ params value from each layer
        data = [[i, [[p, getattr(layer, p, None)] for p in params]] for i, layer in enumerate(model.layersList)]
        # add header
        data = [['Layer#', 'Values']] + data
        # add to logger as InfoTable
        logger.addInfoTable(title, data)

    @staticmethod
    def createForwardStatsInfoTable(model, logger):
        stats = [[i, layer.forwardStats] for i, layer in enumerate(model.layersList) if layer.forwardStats]
        stats = [['Layer#', 'Stats']] + stats

        # return table code
        return logger.createInfoTable('Show', stats)

    # returns InfoTable of model bitwidths
    @staticmethod
    def createBitwidthsTable(model, logger, bitwidthKey):
        # collect model layers bitwidths groups
        table = [[i, [[bitwidthKey, '#Filters']] + layer.getCurrentBitwidth()] for i, layer in enumerate(model.layersList)]
        table.insert(0, ['Layer #', bitwidthKey])
        # create InfoTable
        return logger.createInfoTable(bitwidthKey, table)

    def trainAlphas(self, search_queue, model, architect, nEpoch, loggers):
        print('*** trainAlphas ***')
        loss_container = AvgrageMeter()
        crossEntropy_container = AvgrageMeter()
        bopsLoss_container = AvgrageMeter()

        modelReplicator = architect.modelReplicator

        model.train()

        trainLogger = loggers.get('train')
        # init updateModelWeights() logger func
        loggerFunc = []
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Alphas'.format(nEpoch), self.colsTrainAlphas)
            loggerFunc = [lambda msg: trainLogger.addInfoToDataTable(msg), lambda msg: modelReplicator.logWeightsUpdateMsg(msg, nEpoch)]

        # update model replications weights
        modelReplicator.updateModelWeights(model, loggerFuncs=loggerFunc)

        nBatches = len(search_queue)

        # init logger functions
        def createInfoTable(dict, key, logger, rows):
            dict[key] = logger.createInfoTable('Show', rows)

        def alphasFunc(k, rows):
            createInfoTable(dataRow, self.alphasTableTitle, trainLogger, rows)

        def forwardCountersFunc(rows):
            createInfoTable(dataRow, self.forwardCountersKey, trainLogger, rows)

        for step, (input, target) in enumerate(search_queue):
            startTime = time()
            n = input.size(0)
            dataRow = {}

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            loss, crossEntropyLoss, bopsLoss = architect.step(model, input, target)

            # add alphas data to statistics
            model.stats.addBatchData(model, nEpoch, step)

            func = []
            if trainLogger:
                # log dominant QuantizedOp in each layer
                logDominantQuantizedOp(model, k=self.k, loggerFuncs=[alphasFunc])
                func = [forwardCountersFunc]
            # log forward counters. if loggerFuncs==[] then it is just resets counters
            logForwardCounters(model, loggerFuncs=func)
            # save alphas to csv
            model.save_alphas_to_csv(data=[nEpoch, step])
            # # log allocations
            # self.logAllocations()
            # save loss to container
            loss_container.update(loss, n)
            crossEntropy_container.update(crossEntropyLoss, n)
            bopsLoss_container.update(bopsLoss, n)

            endTime = time()

            # send email
            diffTime = endTime - self.lastMailTime
            if (diffTime > self.secondsBetweenMails) or ((step + 1) % int(nBatches / 2) == 0):
                self.sendEmail(nEpoch, step, nBatches)
                # update last email time
                self.lastMailTime = time()
                # from now on we send every 5 hours
                self.secondsBetweenMails = 5 * 3600

            if trainLogger:
                # calc partition bops ratio
                model.setFiltersByAlphas()
                bopsRatio = model.calcBopsRatio()
                # add missing keys
                dataRow.update({
                    self.batchNumKey: '{}/{}'.format(step, nBatches), self.timeKey: endTime - startTime, self.archLossKey: loss,
                    self.crossEntropyKey: crossEntropyLoss, self.bopsLossKey: bopsLoss, self.pathBopsRatioKey: bopsRatio
                })
                # apply formats
                self._applyFormats(dataRow)
                # add row to data table
                trainLogger.addDataRow(dataRow)

        # log accuracy, loss, etc.
        summaryData = {self.epochNumKey: nEpoch, self.lrKey: architect.lr, self.batchNumKey: 'Summary', self.archLossKey: loss_container.avg,
                       self.crossEntropyKey: crossEntropy_container.avg, self.bopsLossKey: bopsLoss_container.avg}
        self._applyFormats(summaryData)

        for _, logger in loggers.items():
            logger.addSummaryDataRow(summaryData)

        return summaryData

    def trainWeights(self, optimizer, epoch, loggers):
        print('*** trainWeights() ***')
        loss_container = AvgrageMeter()
        top1 = AvgrageMeter()

        model = self.model
        crit = self.cross_entropy
        train_queue = self.train_queue
        grad_clip = self.args.grad_clip

        trainLogger = loggers.get('train')
        if trainLogger:
            self.addModelUNIQstatusTable(model, trainLogger, 'UNIQ status - pre-training weights')
            trainLogger.createDataTable('Epoch:[{}] - Training weights'.format(epoch), self.colsTrainWeights)

        model.train()

        nBatches = len(train_queue)

        for step, (input, target) in enumerate(train_queue):
            startTime = time()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            # choose model partition if we haven't set partition to model
            if self.args.partition is None:
                model.choosePathByAlphas()
            # optimize model weights
            optimizer.zero_grad()
            logits = model(input)
            # calc loss
            loss = crit(logits, target)
            # back propagate
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            # update weights
            optimizer.step()

            prec1 = accuracy(logits, target)[0]
            loss_container.update(loss.item(), n)
            top1.update(prec1.item(), n)

            endTime = time()

            if trainLogger:
                dataRow = {
                    self.batchNumKey: '{}/{}'.format(step, nBatches), self.pathBopsRatioKey: model.calcBopsRatio(),
                    self.bitwidthKey: self.createBitwidthsTable(model, trainLogger, self.bitwidthKey),
                    self.timeKey: (endTime - startTime), self.trainLossKey: loss, self.trainAccKey: prec1
                }
                # if (step + 1) % 20 == 0:
                #     dataRow[self.statsKey] = self.createForwardStatsInfoTable(model, trainLogger)
                # apply formats
                self._applyFormats(dataRow)
                # add row to data table
                trainLogger.addDataRow(dataRow)

        # log accuracy, loss, etc.
        summaryData = {self.trainLossKey: loss_container.avg, self.trainAccKey: top1.avg, self.batchNumKey: 'Summary'}
        # apply formats
        self._applyFormats(summaryData)

        for _, logger in loggers.items():
            logger.addSummaryDataRow(summaryData)

        # log dominant QuantizedOp in each layer
        if trainLogger:
            logDominantQuantizedOp(model, k=self.k,
                                   loggerFuncs=[lambda k, rows: trainLogger.addInfoTable(title=self.alphasTableTitle.format(k), rows=rows)])

        # log forward counters. if loggerFuncs==[] then it is just resets counters
        func = [lambda rows: trainLogger.addInfoTable(title=self.forwardCountersKey, rows=rows)] if trainLogger else []
        logForwardCounters(model, loggerFuncs=func)

        return summaryData

    def infer(self, setModelPartitionFunc, nEpoch, loggers):
        print('*** infer() ***')
        objs = AvgrageMeter()
        top1 = AvgrageMeter()

        model = self.model
        valid_queue = self.valid_queue
        crit = self.cross_entropy

        model.eval()
        # print eval layer index selection
        trainLogger = loggers.get('train')
        if trainLogger:
            trainLogger.createDataTable('Epoch:[{}] - Validation'.format(nEpoch), self.colsValidation)
            # trainLogger.addInfoToDataTable('Layers optimal indices:{}'.format([layer.curr_alpha_idx for layer in model.layersList]))

        nBatches = len(valid_queue)

        # quantize model layers that haven't switched stage yet
        # no need to turn gradients off, since with no_grad() does it
        if model.nLayersQuantCompleted < model.nLayers():
            # turn off noise if 1st unstaged layer
            layer = model.layersList[model.nLayersQuantCompleted]
            layer.turnOffNoise(model.nLayersQuantCompleted)
            # quantize all unstaged layers
            for layerIdx, layer in enumerate(model.layersList[model.nLayersQuantCompleted:]):
                # quantize
                layer.quantize(model.nLayersQuantCompleted + layerIdx)

        # log UNIQ status after quantizing all layers
        self.addModelUNIQstatusTable(model, trainLogger, 'UNIQ status - quantizated for validation')

        # choose model partition if we haven't set partition to model
        if self.args.partition is None:
            setModelPartitionFunc(loggerFuncs=[lambda msg: trainLogger.addInfoToDataTable(msg)])
        # calculate its bops
        bopsRatio = model.calcBopsRatio()

        with no_grad():
            for step, (input, target) in enumerate(valid_queue):
                startTime = time()

                input = Variable(input).cuda()
                target = Variable(target).cuda(async=True)

                logits = model(input)
                loss = crit(logits, target)

                prec1 = accuracy(logits, target)[0]
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)

                endTime = time()

                if trainLogger:
                    dataRow = {
                        self.batchNumKey: '{}/{}'.format(step, nBatches), self.validLossKey: loss, self.validAccKey: prec1,
                        self.timeKey: endTime - startTime
                    }
                    # if (step + 1) % 20 == 0:
                    #     dataRow[self.statsKey] = self.createForwardStatsInfoTable(model, trainLogger)
                    # apply formats
                    self._applyFormats(dataRow)
                    # add row to data table
                    trainLogger.addDataRow(dataRow)

        # restore weights (remove quantization) of model layers that haven't switched stage yet
        if model.nLayersQuantCompleted < model.nLayers():
            for layerIdx, layer in enumerate(model.layersList[model.nLayersQuantCompleted:]):
                # remove quantization
                layer.unQuantize(model.nLayersQuantCompleted + layerIdx)
            # add noise back to 1st unstaged layer
            layer = model.layersList[model.nLayersQuantCompleted]
            layer.turnOnNoise(model.nLayersQuantCompleted)

        # log UNIQ status after restoring model state
        self.addModelUNIQstatusTable(model, trainLogger, 'UNIQ status - state restored')

        # create summary row
        summaryRow = {self.batchNumKey: 'Summary', self.validLossKey: objs.avg, self.validAccKey: top1.avg, self.validBopsRatioKey: bopsRatio}
        # apply formats
        self._applyFormats(summaryRow)

        for _, logger in loggers.items():
            logger.addSummaryDataRow(summaryRow)

        # log forward counters. if loggerFuncs==[] then it is just resets counters
        func = []
        forwardCountersData = [[]]
        if trainLogger:
            func = [lambda rows: forwardCountersData.append(trainLogger.createInfoTable('Show', rows))]

        logForwardCounters(model, loggerFuncs=func)

        if trainLogger:
            # create new data table for validation statistics
            trainLogger.createDataTable('Validation statistics', self.colsValidationStatistics)
            # add bitwidth & forward counters statistics
            dataRow = {
                self.bitwidthKey: self.createBitwidthsTable(model, trainLogger, self.bitwidthKey),
                self.forwardCountersKey: forwardCountersData[-1], self.validBopsRatioKey: bopsRatio
            }
            # apply formats
            self._applyFormats(dataRow)
            # add row to table
            trainLogger.addDataRow(dataRow)
            # add bitwidth to summary row
            summaryRow[self.bitwidthKey] = dataRow[self.bitwidthKey]

        return top1.avg, objs.avg, summaryRow

    def logAllocations(self):
        logger = HtmlLogger(self.args.save, 'allocations', overwrite=True)
        allocationKey = 'Allocation'
        nBatchesKey = 'Number of batches'
        logger.createDataTable('Allocations', [allocationKey, nBatchesKey])

        for bitwidth, nBatches in self.optModelBitwidthCounter.items():
            logger.addDataRow({allocationKey: bitwidth, nBatchesKey: nBatches})

            # bitwidthList = bitwidth[1:-1].replace('),', ');')
            # bitwidthList = bitwidthList.split('; ')
            # bitwidthStr = ''
            # for layerIdx, b in enumerate(bitwidthList):
            #     bitwidthStr += 'Layer [{}]: {}\n'.format(layerIdx, b)
            #
            # logger.addDataRow({allocationKey: bitwidthStr, nBatchesKey: nBatches})

# @staticmethod
# def __getBitwidthKey(optModel_bitwidth):
#     return '{}'.format(optModel_bitwidth)

# def setOptModelBitwidth(self):
#     model = self.model
#     args = self.args
#
#     args.optModel_bitwidth = [layer.getCurrentBitwidth() for layer in model.layersList]
#     # check if current bitwidth has already been sent for training
#     bitwidthKey = self.__getBitwidthKey(args.optModel_bitwidth)
#
#     return bitwidthKey

# # wait for sending all queued jobs
# def waitForQueuedJobs(self):
#     while len(self.optModelTrainingQueue) > 0:
#         self.logger.addInfoToDataTable('Waiting for queued jobs, queue size:[{}]'.format(len(self.optModelTrainingQueue)))
#         self.trySendQueuedJobs()
#         sleep(60)

# # try to send more queued jobs to server
# def trySendQueuedJobs(self):
#     args = self.args
#
#     trySendJobs = len(self.optModelTrainingQueue) > 0
#     while trySendJobs:
#         # take last job in queue
#         command, optModel_bitwidth = self.optModelTrainingQueue[-1]
#         # update optModel_bitwidth in args
#         args.optModel_bitwidth = optModel_bitwidth
#         # get key
#         bitwidthKey = self.__getBitwidthKey(optModel_bitwidth)
#         # save args to JSON
#         saveArgsToJSON(args)
#         # send job
#         retVal = system(command)
#
#         if retVal == 0:
#             # delete the job we sent from queue
#             self.optModelTrainingQueue = self.optModelTrainingQueue[:-1]
#             print('sent model with allocation:{}, queue size:[{}]'
#                   .format(bitwidthKey, len(self.optModelTrainingQueue)))
#
#         # update loop flag, keep sending if current job sent successfully & there are more jobs to send
#         trySendJobs = (retVal == 0) and (len(self.optModelTrainingQueue) > 0)

# def sendOptModel(self, bitwidthKey, nEpoch, nBatch):
#     args = self.args
#
#     # check if this is the 1st time this allocation is optimal
#     if bitwidthKey not in self.optModelBitwidthCounter:
#         self.optModelBitwidthCounter[bitwidthKey] = 0
#
#     # increase model allocation counter
#     self.optModelBitwidthCounter[bitwidthKey] += 1
#
#     # if this allocation has been optimal enough batches, let's train it
#     if self.optModelBitwidthCounter[bitwidthKey] == self.nBatchesOptModel:
#         # save args to JSON
#         saveArgsToJSON(args)
#         # init args JSON destination path on server
#         dstPath = '/home/yochaiz/F-BANNAS/cnn/optimal_models/{}/{}-[{}-{}].json'.format(args.model, args.folderName, nEpoch, nBatch)
#         # init copy command & train command
#         copyJSONcommand = 'scp {} yochaiz@132.68.39.32:{}'.format(args.jsonPath, dstPath)
#         trainOptCommand = 'ssh yochaiz@132.68.39.32 sbatch /home/yochaiz/DropDarts/cnn/sbatch_opt.sh --data {}'.format(dstPath)
#         # perform commands
#         print('%%%%%%%%%%%%%%')
#         print('sent model with allocation:{}, queue size:[{}]'
#               .format(bitwidthKey, len(self.optModelTrainingQueue)))
#         command = '{} && {}'.format(copyJSONcommand, trainOptCommand)
#         retVal = system(command)
#
#         if retVal != 0:
#             # server is full with jobs, add current job to queue
#             self.optModelTrainingQueue.append((command, args.optModel_bitwidth))
#             print('No available GPU, adding {} to queue, queue size:[{}]'
#                   .format(bitwidthKey, len(self.optModelTrainingQueue)))
#             # remove args JSON
#             system('ssh yochaiz@132.68.39.32 rm {}'.format(dstPath))
#
#     # try to send queued jobs, regardless current optimal model
#     self.trySendQueuedJobs()

# def trainOptimalModel(self, nEpoch, nBatch):
#     model = self.model
#     # set optimal model bitwidth per layer
#     optBopsRatio = model.evalMode()
#     bitwidthKey = self.setOptModelBitwidth()
#     # train optimal model
#     self.sendOptModel(bitwidthKey, nEpoch, nBatch)
#
#     return optBopsRatio
# # ======================================
# # set model partition
# model.choosePathByAlphas()
# # save model layers partition
# args.partition = model.getCurrentFiltersPartition()
# # save args to checkpoint
# from torch import save as saveCheckpoint
# checkpointPath = '{}/chkpnt.json'.format(args.save)
# saveCheckpoint(args, checkpointPath)
# # init args JSON destination path on server
# jsonFileName = '{}-{}-[{}].json'.format(args.folderName, 0, 0)
# dstPath = '/home/vista/Desktop/Architecture_Search/F-BANNAS/cnn/trained_models/{}/{}/{}'.format(args.model, args.dataset, jsonFileName)
# from shutil import copy
# copy(checkpointPath, dstPath)
# from cnn.train_opt2 import G
# t = dict(data=dstPath, gpu=[0])
# from argparse import Namespace
# t = Namespace(**t)
# G(t)
# # reset args.alphas
# args.alphas = None
# # =====================================

# # ==============================================================================
# from torch import tensor, IntTensor
# partition = None
# if args.partition == 1:
# partition = [[0, 0, 3, 7, 6], [2, 8, 1, 1, 4], [6, 2, 2, 2, 4], [0, 0, 7, 4, 5], [1, 13, 0, 0, 2], [5, 2, 2, 5, 2], [0, 0, 7, 6, 3],
#              [7, 12, 4, 2, 7], [27, 1, 1, 3], [18, 5, 2, 2, 5], [0, 0, 0, 29, 3], [4, 19, 3, 1, 5], [1, 1, 2, 6, 22],
#              [1, 0, 14, 11, 6], [3, 3, 53, 2, 3], [1, 0, 0, 63], [3, 46, 7, 5, 3], [1, 0, 0, 61, 2], [2, 57, 1, 2, 2],
#              [58, 3, 1, 0, 2], [32, 20, 0, 3, 9]]
# partition = [[0, 1, 12, 3, 0], [4, 9, 1, 1, 1], [1, 0, 10, 3, 2], [3, 1, 11, 0, 1], [3, 9, 2, 2, 0], [4, 2, 3, 7, 0], [1, 0, 10, 4, 1],
#              [7, 11, 4, 8, 2], [26, 5, 1, 0], [18, 6, 2, 6, 0], [1, 2, 1, 28, 0], [3, 25, 0, 2, 2], [5, 2, 14, 6, 5], [0, 2, 22, 4, 4],
#              [5, 1, 51, 5, 2], [22, 0, 4, 38], [12, 6, 26, 17, 3], [9, 0, 1, 52, 2], [26, 3, 12, 22, 1], [60, 4, 0, 0, 0],
#              [38, 9, 0, 9, 8]]
# elif args.partition == 2:
#     # partition = [[0, 0, 5, 7, 4], [4, 10, 1, 0, 1], [9, 1, 3, 2, 1], [1, 0, 10, 4, 1], [1, 13, 1, 1, 0], [3, 1, 4, 7, 1], [1, 0, 4, 9, 2],
#     #              [9, 13, 5, 1, 4], [30, 0, 0, 2], [16, 7, 2, 3, 4], [2, 0, 0, 28, 2], [3, 20, 8, 0, 1], [0, 0, 4, 10, 18],
#     #              [1, 1, 15, 11, 4], [3, 3, 54, 3, 1], [2, 1, 0, 61], [6, 46, 10, 2, 0], [0, 0, 0, 64, 0], [1, 60, 1, 1, 1],
#     #              [57, 4, 1, 2, 0], [32, 23, 0, 3, 6]]
#     partition = [[0, 0, 14, 2, 0], [0, 11, 1, 3, 1], [2, 3, 3, 4, 4], [3, 1, 8, 3, 1], [3, 12, 1, 0, 0], [4, 4, 5, 3, 0], [1, 1, 10, 2, 2],
#              [6, 11, 6, 5, 4], [22, 4, 4, 2], [26, 3, 1, 1, 1], [1, 1, 3, 26, 1], [3, 21, 1, 2, 5], [6, 5, 12, 6, 3], [1, 1, 18, 8, 4],
#                  [6, 2, 55, 0, 1], [19, 1, 5, 39], [14, 6, 25, 18, 1], [14, 2, 2, 45, 1], [16, 9, 14, 21, 4], [62, 2, 0, 0, 0],
#                  [45, 9, 0, 5, 5]]
#
# for i in range(len(partition)):
#     partition[i] = tensor(partition[i]).type(IntTensor)
#
# for p, layer in zip(partition, model.layersList):
#     print(p)
#     assert (p.sum() == layer.nFilters())
#     assert (len(p) == layer.numOfOps())
#     layer.setFiltersPartition(p)
# # ==============================================================================

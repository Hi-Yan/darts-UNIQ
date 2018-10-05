from torch.optim import SGD

from .regime import TrainRegime, initTrainLogger, save_checkpoint, HtmlLogger
from cnn.architect import Architect
import cnn.gradEstimators as gradEstimators


class AlphasWeightsLoop(TrainRegime):
    def __init__(self, args, model, modelClass, logger):
        super(AlphasWeightsLoop, self).__init__(args, model, modelClass, logger)

        # init model replicator
        replicatorClass = gradEstimators.__dict__[args.grad_estimator]
        replicator = replicatorClass(model, modelClass, args)
        # init architect
        self.architect = Architect(replicator, args)

    def train(self):
        model = self.model
        args = self.args
        logger = self.logger
        # init number of epochs
        nEpochs = self.model.nLayers()
        # init validation best precision value
        best_prec1 = 0.0

        for epoch in range(1, nEpochs + 1):
            # turn on alphas
            model.turnOnAlphas()
            print('========== Epoch:[{}] =============='.format(epoch))
            # init epoch train logger
            trainLogger = HtmlLogger(self.trainFolderPath, str(epoch))
            # set loggers dictionary
            loggersDict = dict(train=trainLogger)
            # train alphas
            alphaData = self.trainAlphas(self.search_queue, model, self.architect, epoch, loggersDict)

            # validation on current optimal model
            valid_acc, validData = self.infer(epoch, loggersDict)

            # merge trainData with validData
            for k, v in validData.items():
                alphaData[k] = v

            # save model checkpoint
            is_best = valid_acc > best_prec1
            best_prec1 = max(valid_acc, best_prec1)
            save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best)

            # add data to main logger table
            logger.addDataRow(alphaData)

            ## train weights ##
            # create epoch train weights folder
            epochName = '{}_w'.format(epoch)
            epochFolderPath = '{}/{}'.format(self.trainFolderPath, epochName)
            # turn off alphas
            model.turnOffAlphas()
            # turn on weights gradients
            model.turnOnWeights()
            # init optimizer
            optimizer = SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            # train weights with 1 epoch per stage
            wEpoch = 1
            switchStageFlag = True
            while switchStageFlag:
                # init epoch train logger
                trainLogger = HtmlLogger(epochFolderPath, '{}_{}'.format(epochName, wEpoch))
                # train stage weights
                self.trainWeights(model.choosePathByAlphas, optimizer, wEpoch, dict(train=trainLogger))
                # switch stage
                switchStageFlag = model.switch_stage([lambda msg: trainLogger.addInfoToDataTable(msg)])
                # update epoch number
                wEpoch += 1

            # init epoch train logger for last epoch
            trainLogger = HtmlLogger(epochFolderPath, '{}_{}'.format(epochName, wEpoch))
            # set loggers dictionary
            loggersDict = dict(train=trainLogger)
            # last weights training epoch we want to log also to main logger
            trainData = self.trainWeights(model.choosePathByAlphas, optimizer, wEpoch, loggersDict)
            # validation on optimal model
            valid_acc, validData = self.infer(wEpoch, loggersDict)

            # update epoch
            trainData[self.epochNumKey] = epoch
            # merge trainData with validData
            for k, v in validData.items():
                trainData[k] = v

            # save model checkpoint
            is_best = valid_acc > best_prec1
            best_prec1 = max(valid_acc, best_prec1)
            save_checkpoint(self.trainFolderPath, model, args, epoch, best_prec1, is_best)

            # add data to main logger table
            logger.addDataRow(trainData)

        # send final email
        self.sendEmail('Final', 0, 0)

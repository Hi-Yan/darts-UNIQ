from .regime import TrainRegime, trainAlphas, infer
from cnn.utils import initTrainLogger, save_checkpoint
from cnn.architect import Architect


class AlphasOnly(TrainRegime):
    def __init__(self, args, model, modelClass,logger):
        super(AlphasOnly, self).__init__(args, model, modelClass, logger)

        # init architect
        self.architect = Architect(model, modelClass, args)

    def train(self):
        # init number of epochs
        nEpochs = self.model.nLayers()
        # init validation best precision value
        best_prec1 = 0.0
        # turn on alphas
        self.model.turnOnAlphas()

        for epoch in range(self.epoch + 1, self.epoch + nEpochs + 1):
            print('========== Epoch:[{}] =============='.format(epoch))
            # init epoch train logger
            trainLogger = initTrainLogger(str(epoch), self.trainFolderPath, self.args.propagate)
            # set loggers dictionary
            loggersDict = dict(train=trainLogger, main=self.logger)
            # train alphas
            trainAlphas(self.search_queue, self.model, self.architect, epoch, loggersDict)
            # validation on current optimal model
            valid_acc = infer(self.valid_queue, self.model, self.model.evalMode, self.cross_entropy, epoch, loggersDict)


            # update architecture learning rate
            if self.updateLR == 1:
                self.architect.lr = max(self.architect.lr / 10, 0.001)
            self.updateLR = (self.updateLR + 1) % 2

            # save model checkpoint
            is_best = valid_acc > best_prec1
            best_prec1 = max(valid_acc, best_prec1)
            save_checkpoint(self.trainFolderPath, self.model, epoch, best_prec1, is_best)
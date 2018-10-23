from os import system, remove
from os.path import exists
from multiprocessing import Pool
from time import sleep

from torch.optim import SGD
from torch import save as saveCheckpoint
from torch import load as loadCheckpoint

from .regime import TrainRegime, save_checkpoint, HtmlLogger
from cnn.architect import Architect
import cnn.gradEstimators as gradEstimators


def escape(txt):
    translation = str.maketrans({'[': '\[', ']': '\]', '(': '\(', ')': '\)', ',': '\,', ' ': '\ '})
    return txt.translate(translation)


def printSuccessOrFail(retVal, msg):
    res = 'Success' if retVal == 0 else 'Fail'
    print(msg.format(res))


def manageJobs(epochJobs, epoch):
    # copy jobs JSON to server
    for job in epochJobs:
        # perform command
        retVal = system(job.cmdCopyToServer)
        printSuccessOrFail(retVal, 'Copied +{}+ to server'.format(job.jsonFileName) + ':[{}]')
    # init server names
    servers = ['gaon6', 'gaon4', 'gaon2', 'gaon5']
    # init maximal SBATCH jobs we can run concurrently
    nMaxSBATCH = 4
    # init number of maximal GPUs we can run in single sbatch
    nMaxGPUs = 2
    # init number of maximal CPUs
    nMaxCPUs = 10
    # init number of minutes to wait
    nMinsWaiting = 10
    # try sending as much jobs as possible under single sbatch
    # try sending as much sbatch commands as possible
    while len(epochJobs) > 0:
        print('Epoch:[{}] Jobs waiting:[{}]'.format(epoch, len(epochJobs)))
        # set number of jobs we want to run in a single SBATCH command
        nJobs = min(nMaxGPUs, len(epochJobs))
        # try to send sbatch command to server, stop when successful
        retVal = -1
        while (nJobs > 0) and (retVal != 0):
            # concatenate JSON files for jobs
            files = ''
            for job in epochJobs[:nJobs]:
                files += escape(job.dstPath) + '?'
            # remove last comma
            files = files[:-1]
            # set number of CPUs
            nCPUs = min(nMaxCPUs, 3 * nJobs)
            # count how many running SBATCH jobs we have on server
            nRunningJobs = system('ssh yochaiz@132.68.39.32 ./countRunning.sh') >> 8
            nPendingJobs = system('ssh yochaiz@132.68.39.32 ./countPending.sh') >> 8
            print('Epoch:[{}] - nRunningJobs:[{}] nPendingJobs:[{}]'.format(epoch, nRunningJobs, nPendingJobs))
            # if there is room left for running a job, try to run it
            if nRunningJobs < nMaxSBATCH:
                # change servers order if there are pending jobs
                if nPendingJobs > 0:
                    s = servers[0]
                    servers = servers[1:]
                    servers.append(s)
                # try to perform command on one of the servers
                for serv in servers:
                    print('Epoch:[{}] - trying to send [{}] trainings to [{}], jobs still waiting:[{}]'.format(epoch, nJobs, serv, len(epochJobs)))
                    # create command
                    trainCommand = 'ssh yochaiz@132.68.39.32 sbatch --gres=gpu:{} -c {} -w {} /home/yochaiz/F-BANNAS/cnn/sbatch_opt.sh --data "{}"' \
                        .format(nJobs, nCPUs, serv, files)
                    retVal = system(trainCommand)
                    # clear successfully sent jobs
                    if retVal == 0:
                        epochJobs = epochJobs[nJobs:]
                        print('Epoch:[{}] - sent [{}] trainings successfully to [{}], jobs still waiting:[{}]'
                              .format(epoch, nJobs, serv, len(epochJobs)))
                        break

            # check if jobs not sent, try sending less jobs, i.e. use less GPUs
            # we don't really need to check retVal here, but anyway ...
            if retVal != 0:
                nJobs -= 1

            sleep(10)

        # if didn't manage to send any job, wait 10 mins
        if retVal != 0:
            print('Epoch:[{}] - did not manage to send trainings,  waiting [{}] mins'.format(epoch, nMinsWaiting))
            # sleep(nMinsWaiting * 60)
            sleep(10)

    print('Epoch:[{}] - sent all jobs successfully'.format(epoch))
    print('Epoch:[{}] - Done !'.format(epoch))


class TrainingJob:
    def __init__(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)

        # set dstPath
        self.dstPath = '{}/{}'.format(self.remoteDirPath, self.jsonFileName)
        # init copy to server command
        self.cmdCopyToServer = 'scp {} "yochaiz@132.68.39.32:{}"'.format(escape(self.jsonPath), escape(self.dstPath))
        # init copy from server command
        self.cmdCopyFromServer = 'scp "yochaiz@132.68.39.32:{}" "{}"'.format(escape(self.dstPath), self.jsonPath)


class AlphasWeightsLoop(TrainRegime):
    def __init__(self, args, logger):
        super(AlphasWeightsLoop, self).__init__(args, logger)

        # set number of different partitions we want to draw from alphas multinomial distribution in order to estimate their validation accuracy
        self.nValidPartitions = 3
        # create dir on remove server
        self.remoteDirPath = '/home/yochaiz/F-BANNAS/cnn/trained_models/{}/{}/{}'.format(args.model, args.dataset, args.folderName)
        command = 'ssh yochaiz@132.68.39.32 mkdir "{}"'.format(escape(self.remoteDirPath))
        retVal = system(command)
        printSuccessOrFail(retVal, 'Created folder [{}] on remote server'.format(self.remoteDirPath) + ':[{}]')

        # init list of training jobs we yet have to get their values
        self.jobsList = []
        # init data table row keys to replace
        self.rowKeysToReplace = [self.validLossKey, self.validAccKey]
        # init validation best precision value from all training jobs
        self.best_prec1 = 0.0

        # ========================== DEBUG ===============================
        # return
        # ================================================================

        # init model replicator
        replicatorClass = gradEstimators.__dict__[args.grad_estimator]
        replicator = replicatorClass(self.model, self.modelClass, args, logger)
        # init architect
        self.architect = Architect(replicator, args)

    # run on validation set and add validation data to main data row
    def __inferWithData(self, setModelPathFunc, epoch, loggersDict, dataRow):
        # run on validation set
        _, _, validData = self.infer(setModelPathFunc, epoch, loggersDict)

        # update epoch
        dataRow[self.epochNumKey] = epoch
        # merge dataRow with validData
        for k, v in validData.items():
            dataRow[k] = v

    # creates job of training model partition
    def __createTrainingJob(self, setModelPartitionFunc, nEpoch, id):
        args = self.args
        model = self.model
        # set model partition
        setModelPartitionFunc()
        # set JSON file name
        jsonFileName = '{}-{}-[{}].json'.format(args.folderName, nEpoch, id)
        # create training job instance
        trainingJob = TrainingJob(dict(bopsRatio=model.calcBopsRatio(), bops=model.countBops(), remoteDirPath=self.remoteDirPath,
                                       bitwidthInfoTable=self.createBitwidthsTable(model, self.logger, self.bitwidthKey),
                                       jsonFileName=jsonFileName, jsonPath='{}/{}'.format(args.save, jsonFileName)))

        # save model layers partition
        args.partition = model.getCurrentFiltersPartition()
        # save args to checkpoint
        saveCheckpoint(args, trainingJob.jsonPath)
        # reset args.partition
        args.partition = None

        return trainingJob

    # create all training jobs for a single epoch
    def __createEpochJobs(self, epoch):
        model = self.model
        # train from scratch (from 32-bit pre-trained) model partitions based on alphas distribution
        epochJobs = []

        # single training on setFiltersByAlphas
        epochJobs.append(self.__createTrainingJob(model.setFiltersByAlphas, epoch, 0))
        # nValidPartitions trainings on setFiltersByAlphas
        for id in range(1, self.nValidPartitions + 1):
            epochJobs.append(self.__createTrainingJob(model.choosePathByAlphas, epoch, id))

        # create process to manage epoch jobs
        pool = Pool(processes=1)
        pool.apply_async(manageJobs, args=(epochJobs, epoch,))

        return epochJobs

    def train(self):
        model = self.model
        args = self.args
        logger = self.logger
        # init number of epochs
        nEpochs = self.model.nLayers()

        # # ========================== DEBUG ===============================
        # # create epoch jobs
        # for epoch in range(1, 4):
        #     epochJobsList = self.__createEpochJobs(epoch)
        # # ================================================================

        for epoch in range(1, nEpochs + 1):
            print('========== Epoch:[{}] =============='.format(epoch))
            # init epoch train logger
            trainLogger = HtmlLogger(self.trainFolderPath, str(epoch))
            # set loggers dictionary
            loggersDict = dict(train=trainLogger)

            # train alphas
            dataRow = self.trainAlphas(self.search_queue[epoch % args.alphas_data_parts], model, self.architect, epoch, loggersDict)

            # create epoch jobs
            epochJobsList = self.__createEpochJobs(epoch)
            # merge current epoch JSONs with the rest of JSONs
            self.jobsList.extend(epochJobsList)

            # validation on fixed partition by alphas values
            self.__inferWithData(model.setFiltersByAlphas, epoch, loggersDict, dataRow)
            # add data to main logger table
            logger.addDataRow(dataRow)

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
                self.trainWeights(optimizer, wEpoch, dict(train=trainLogger))
                # switch stage
                switchStageFlag = model.switch_stage([lambda msg: trainLogger.addInfoToDataTable(msg)])
                # update epoch number
                wEpoch += 1

            # init epoch train logger for last epoch
            trainLogger = HtmlLogger(epochFolderPath, '{}_{}'.format(epochName, wEpoch))
            # set loggers dictionary
            loggersDict = dict(train=trainLogger)
            # last weights training epoch we want to log also to main logger
            dataRow = self.trainWeights(optimizer, wEpoch, loggersDict)

            # validation on fixed partition by alphas values
            self.__inferWithData(model.setFiltersByAlphas, epoch, loggersDict, dataRow)
            # add data to main logger table
            logger.addDataRow(dataRow)

            # add data rows for epoch JSONs
            self.__addEpochJSONsDataRows(epochJobsList, epoch)

            # update temp values in data table + update bops plot
            self.__updateDataTableAndBopsPlot()

            # save checkpoint
            save_checkpoint(self.trainFolderPath, model, args, epoch, self.best_prec1)

        # send final email
        self.sendEmail('Final', 0, 0)

    @staticmethod
    def generateTempValue(jsonFileName, key):
        return '{}_{}'.format(jsonFileName, key)

    def __updateDataTableAndBopsPlot(self):
        # init plot data list
        bopsPlotData = []
        best_prec1 = 0.0
        # init updated jobs list, a list of jobs we haven't got their values yet
        updatedJobsList = []
        # copy files back from server and check if best_prec1, best_valid_loss exists
        for job in self.jobsList:
            # copy JSON from server back here
            retVal = system(job.cmdCopyFromServer)
            printSuccessOrFail(retVal, 'Copied +{}+ back from server'.format(job.jsonFileName) + ':[{}]')
            # load checkpoint
            if exists(job.jsonPath):
                checkpoint = loadCheckpoint(job.jsonPath, map_location=lambda storage, loc: storage.cuda())
                # init list of existing keys we found in checkpoint
                existingKeys = []
                # update keys if they exist
                for key in self.rowKeysToReplace:
                    v = getattr(checkpoint, key, None)
                    # if v is not None, then update value in table
                    if v is not None:
                        # update key exists
                        existingKeys.append(key)
                        # replace value in table
                        self.logger.replaceValueInDataTable(self.generateTempValue(job.jsonFileName, key), self.formats[key].format(v))
                        # add tuple of (bitwidth, bops, accuracy) to plotData if we have accuracy value
                        if key == self.validAccKey:
                            bopsPlotData.append((None, job.bops, v))
                            # update best_prec1 of all training jobs we have trained
                            self.best_prec1 = max(self.best_prec1, v)

                # add job to updatedJobsList if we haven't got all keys, otherwise we are done with this job
                if len(existingKeys) < len(self.rowKeysToReplace):
                    updatedJobsList.append(job)
                # else:
                #     remove(job.jsonPath)

        # update self.jobsList to updatedJobsList
        self.jobsList = updatedJobsList

        # send new bops plot value to plot
        self.model.stats.addBopsData(dict(Alpha=bopsPlotData))

        return best_prec1

    def __addEpochJSONsDataRows(self, epochJobsList, epoch):
        logger = self.logger

        for job in epochJobsList:
            # set data row unique values for json, in order to overwrite them in the future when we will have values
            dataRow = {self.epochNumKey: epoch, self.validBopsRatioKey: job.bopsRatio, self.bitwidthKey: job.bitwidthInfoTable}
            # apply formats
            self._applyFormats(dataRow)
            for key in self.rowKeysToReplace:
                dataRow[key] = self.generateTempValue(job.jsonFileName, key)
            # add data row
            logger.addDataRow(dataRow)

# def train(self):
#     # init logger data table
#     self.logger.createDataTable('Summary', self.colsMainLogger)
#     # init number of epochs
#     nEpochs = self.model.nLayers()
#
#     for epoch in range(1, nEpochs + 1):
#         print('========== Epoch:[{}] =============='.format(epoch))
#         # create epoch jobs
#         epochJobsList = self.__createEpochJobs(epoch)
#         # merge current epoch JSONs with the rest of JSONs
#         self.jobsList.extend(epochJobsList)
#
#         # add data rows for epoch JSONs
#         self.__addEpochJSONsDataRows(epochJobsList, epoch)
#
#         # update temp values in data table + update bops plot
#         self.__updateDataTableAndBopsPlot()

# # create all training jobs for a single epoch
# def __createEpochJobs(self, epoch):
#     model = self.model
#     # train from scratch (from 32-bit pre-trained) model partitions based on alphas distribution
#     epochJobs = []
#     # init json files names list
#     jsonFilesList = []
#     # single training on setFiltersByAlphas
#     epochJobs.append(self.__createTrainingJob(model.setFiltersByAlphas, epoch, 0))
#     # epochJobsData[jsonFileName] = [bopsRatio, bitwidthInfoTable]
#     # jsonFilesList.append([jsonFileName, jsonPath])
#     # nValidPartitions trainings on setFiltersByAlphas
#     for id in range(1, self.nValidPartitions + 1):
#         epochJobs.append(self.__createTrainingJob(model.choosePathByAlphas, epoch, id))
#         # jsonFileName, jsonPath, bopsRatio, bitwidthInfoTable = self.__createTrainingJob(model.choosePathByAlphas, epoch, id)
#         # epochJobsData[jsonFileName] = [bopsRatio, bitwidthInfoTable]
#         # jsonFilesList.append([jsonFileName, jsonPath])
#
#     # init JSON path on server
#     dstPathList = []
#     # init list of commands to copy JSON from server back to here, in order to read its values back to data table
#     copyJSONcmdList = []
#     # copy JSON files to server
#     for jsonFileName, jsonPath in jsonFilesList:
#         # init args checkpoint destination path on server
#         dstPath = '{}/{}'.format(self.remoteDirPath, jsonFileName)
#         dstPathList.append(dstPath)
#         # init copy command
#         cmd = 'scp {} "yochaiz@132.68.39.32:{}"'.format(escape(jsonPath), escape(dstPath))
#         # init reverse copy command, from server back here
#         reverseCommand = 'scp "yochaiz@132.68.39.32:{}" "{}"'.format(escape(dstPath), jsonPath)
#         copyJSONcmdList.append(reverseCommand)
#         # perform command
#         retVal = system(cmd)
#         printSuccessOrFail(retVal, 'Copied +{}+ to server'.format(jsonFileName) + ':[{}]')
#
#     # add reverse copy JSON commands to jsonFilesList
#     for i, cmd in enumerate(copyJSONcmdList):
#         jsonFilesList[i].append(cmd)
#
#     # create process to manage epoch jobs
#     pool = Pool(processes=1)
#     pool.apply_async(manageJobs, args=(epochJobs, epoch,))
#
#     return epochJobsData, jsonFilesList

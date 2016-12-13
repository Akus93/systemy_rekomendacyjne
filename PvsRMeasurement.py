import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import random
import copy

from my_rec_system import MyRecomendationSystem


def getUDataContent():
    plik = open('data/u.data', 'r')
    zawartosc = plik.read()
    zawartosc2 = zawartosc.split('\n')
    zawartosc4 = []
    zbiorUsers = set()
    zbiorMovies = set()
    for wiersz in zawartosc2[:-1]:
        wierszPodzielon = wiersz.split('\t')
        zawartosc4.append(wierszPodzielon)
        zbiorUsers.add(wierszPodzielon[0])
        zbiorMovies.add(wierszPodzielon[1])
    return zawartosc4, zbiorUsers, zbiorMovies


def findTheFirstEVectorBySVD(matrix):
    U, S, VT = np.linalg.svd(matrix)
    return U[:, 0]


def getMatrixSpectrum(matrix):
    U, S, VT = np.linalg.svd(matrix)
    return S


def findTheFirstEVectorByPowerIteration(matrix):
    numberOfIterations = 0
    tempMatrix = np.dot(matrix, np.transpose(matrix))
    tempVector = np.ones(matrix.shape[0])
    oldTempVector = tempVector
    tempVector = np.dot(tempMatrix, tempVector)
    tempVector /= math.sqrt(np.sum(tempVector ** 2))
    while not (np.allclose(tempVector, oldTempVector)):
        numberOfIterations += 1
        oldTempVector = tempVector
        tempVector = np.dot(tempMatrix, tempVector)
        tempVector /= math.sqrt(np.sum(tempVector ** 2))
    return tempVector, numberOfIterations


def findTheFirstEVector(matrix):
    theFirstEVector, numberOfIterations = findTheFirstEVectorByPowerIteration(matrix)
    return theFirstEVector


def findRowAveragesVector(matrix):
    theFirstEVector = np.sum(matrix, axis=0)
    return theFirstEVector


def getSelectedRows(firstEVector, n, mode=0):
    firstEVectorWithSquaredEntriesAsList = list(firstEVector)
    vectorToRank = []
    for squaredEntryNumber in range(len(firstEVectorWithSquaredEntriesAsList)):
        vectorToRank.append([squaredEntryNumber, firstEVectorWithSquaredEntriesAsList[squaredEntryNumber]])
    if mode == 0:
        vectorToRankSorted = sorted(vectorToRank, key=lambda x: x[1], reverse=True)
    if mode == 1:
        vectorToRankSorted = sorted(vectorToRank, key=lambda x: x[1], reverse=False)
    if mode == 2:
        random.shuffle(vectorToRank)
        vectorToRankSorted = vectorToRank

    NSelectedRows = [row[0] for row in vectorToRankSorted]
    return NSelectedRows[:n]


def getSubUDataSet(datasetSizeReductionRatio, mode):
    UDataContent, zbiorUsers, zbiorMovies = getUDataContent()

    tablica3 = np.zeros((len(zbiorUsers), len(zbiorMovies)))
    for wiersz in UDataContent:
        tablica3[int(wiersz[0]) - 1, int(wiersz[1]) - 1] = 1

    numberOfSelectedUsers = int(round(len(zbiorUsers) * datasetSizeReductionRatio))

    numberOfSelectedMovies = int(round(len(zbiorMovies) * datasetSizeReductionRatio))

    if mode == 0:
        U0 = findTheFirstEVector(tablica3)
        V0 = findTheFirstEVector(np.transpose(tablica3))

        selectedUsers = getSelectedRows(U0 ** 2, numberOfSelectedUsers, mode)
        selectedMovies = getSelectedRows(V0 ** 2, numberOfSelectedMovies, mode)

    if mode == 1:
        rowAveragesVector = findRowAveragesVector(tablica3)
        columnAveragesVector = findRowAveragesVector(np.transpose(tablica3))
        selectedUsers = getSelectedRows(rowAveragesVector, numberOfSelectedUsers, 0)
        selectedMovies = getSelectedRows(columnAveragesVector, numberOfSelectedMovies, 0)

    selectedUsersSet = set(selectedUsers)
    selectedMoviesSet = set(selectedMovies)
    subdataset = [wiersz[:3] for wiersz in UDataContent if
                  (int(wiersz[0]) in selectedUsersSet) and (int(wiersz[1]) in selectedMoviesSet)]

    return subdataset


def convertSubUDataSetToInTuplesList(subUDataSet, trueToFalseThreshold):
    inTuplesList = []
    for subUDataSetRow in subUDataSet:
        inTupleRow = []
        if int(subUDataSetRow[2]) > trueToFalseThreshold:
            inTupleRow.append(1)
        else:
            inTupleRow.append(-1)
        inTupleRow.append(subUDataSetRow[0])
        inTupleRow.append(subUDataSetRow[1])
        inTuplesList.append(tuple(inTupleRow))
    return inTuplesList


class DimensionsIndexingDictionaries:
    def __init__(self, listaKrotekWejsciowych=[], numberOfDataAttributes=1):
        if not (listaKrotekWejsciowych == []):
            numberOfDataAttributes = len(listaKrotekWejsciowych[0])
        self.liczbaWymiarowIndeksowaniaKomorekTensoraWeWy = numberOfDataAttributes
        self.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania = []
        self.listaSlownikowIndeksowania = []
        for i in range(self.liczbaWymiarowIndeksowaniaKomorekTensoraWeWy):
            self.listaSlownikowIndeksowania.append({})
            self.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania.append([])
            self.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[i] = 0
        self.update(listaKrotekWejsciowych)

    def update(self, listaKrotekWejsciowych=[]):
        for aktualnaKrotka in listaKrotekWejsciowych:
            aktualnaKrotkaJakoLista = list(aktualnaKrotka)
            for indeksSlownika in range(self.liczbaWymiarowIndeksowaniaKomorekTensoraWeWy):
                aktualnyKlucz = aktualnaKrotkaJakoLista[indeksSlownika]
                if (aktualnyKlucz not in self.listaSlownikowIndeksowania[indeksSlownika].keys()):
                    self.listaSlownikowIndeksowania[indeksSlownika][aktualnyKlucz] = len(
                        self.listaSlownikowIndeksowania[indeksSlownika].keys()) + 1
                    self.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[indeksSlownika] += 1
        return self.listaSlownikowIndeksowania

    def getIndex(self, numerSlownika, kluczAdresujacyIndeks):
        if (kluczAdresujacyIndeks not in self.listaSlownikowIndeksowania[numerSlownika].keys()):
            self.listaSlownikowIndeksowania[numerSlownika][kluczAdresujacyIndeks] = len(
                self.listaSlownikowIndeksowania[numerSlownika].keys()) + 1
            self.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[numerSlownika] += 1
        return self.listaSlownikowIndeksowania[numerSlownika][kluczAdresujacyIndeks]


class ArrayBasedDataRepresentation:
    def __init__(self, dids, trainSetTuplesList=[]):
        dids.update(trainSetTuplesList)
        self.inputDataRepresentationShape = dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[1:]
        self.trainsetDataRepresentationShape = []
        for i in range(len(self.inputDataRepresentationShape)):
            self.trainsetDataRepresentationShape.append(self.inputDataRepresentationShape[i] + 1)
        self.trainsetDataRepresentation = np.zeros((tuple(self.trainsetDataRepresentationShape)))
        self.trainsetDataRepresentation, self.rowSumsVector, self.columnSumsVector = self.storeData(dids,
                                                                                                    trainSetTuplesList)

    def storeData(self, dids, trainSetTuplesList=[]):
        for aktualnaKrotka in trainSetTuplesList:
            macierzTymczasowa = np.zeros((1, 1))
            macierzTymczasowa[0, 0] = aktualnaKrotka[0]

            ksztaltMacierzyTymczasowej = ()
            for i in range(len(dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[1:])):
                wektorTymczasowy = np.zeros(
                    (dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[i + 1] + 1, 1))
                wektorTymczasowy[0] = 1
                wektorTymczasowy[dids.listaSlownikowIndeksowania[i + 1][aktualnaKrotka[i + 1]]] = 1
                ksztaltMacierzyTymczasowej = ksztaltMacierzyTymczasowej + (
                    dids.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[i + 1] + 1,)

                macierzTymczasowa = (np.outer(macierzTymczasowa, wektorTymczasowy)).reshape(ksztaltMacierzyTymczasowej)
            self.trainsetDataRepresentation = np.add(self.trainsetDataRepresentation, macierzTymczasowa)

            self.rowSumsVector = self.trainsetDataRepresentation[0, :]

            self.columnSumsVector = self.trainsetDataRepresentation[:, 0]
        return self.trainsetDataRepresentation, self.rowSumsVector, self.columnSumsVector


class TestSetObject:
    def __init__(self, testSetTuplesList):
        self.sciagaDlaWszechwiedzacego = {}
        self.queryTuplesList = []
        for currentTestTuple in testSetTuplesList:
            currentTestTupleAsList = list(currentTestTuple)
            whitenedCurrentTestTuple = currentTestTupleAsList
            whitenedCurrentTestTuple[0] = 0
            whitenedCurrentTestTuple = tuple(whitenedCurrentTestTuple[1:])
            self.sciagaDlaWszechwiedzacego[whitenedCurrentTestTuple] = currentTestTuple[0]
            self.queryTuplesList.append(whitenedCurrentTestTuple)


class RecSystem:
    def __init__(self, dids1, inputArray, systemType):
        self.typeOfSystem = systemType
        self.trainsetDataRepresentation = inputArray
        self.dids1 = dids1
        self.numberOfDimensionsToBeLeft = min(
            dids1.liczbaWymiarowPrzestrzeniOdpowiadajacejDanemuKierunkowiIndeksowania[1:])
        self.inputDataProcessed = False
        self.sciagaDlaWszechwiedzacego = {}

    def processInputArray(self):
        k = self.numberOfDimensionsToBeLeft

        # my-rec-system
        if self.typeOfSystem == 7:
            # my_rec_system = MyRecomendationSystem(self.trainsetDataRepresentation, 'data/u.data')
            # self.processedDataRepresentation = my_rec_system.get_result()
            self.coreDataRepresentation = self.trainsetDataRepresentation[1:, 1:]

            self.rowSumsVector = self.trainsetDataRepresentation[1:, 0]
            self.columnSumsVector = self.trainsetDataRepresentation[0, 1:]
            self.rowAveragesVector = self.rowSumsVector / len(self.rowSumsVector)
            self.columnAveragesVector = self.columnSumsVector / len(self.columnSumsVector)
            self.columnAveragesMatrix = np.outer(np.ones(len(self.rowSumsVector)), self.columnAveragesVector)
            self.rowAveragesMatrix = np.outer(self.rowAveragesVector, np.ones(len(self.columnSumsVector)))
            self.processedDataRepresentation = self.columnAveragesMatrix + self.rowAveragesMatrix

        if self.typeOfSystem == 3:
            self.rowSumsVector = self.trainsetDataRepresentation[1:, 0]
            self.columnSumsVector = self.trainsetDataRepresentation[0, 1:]
            self.rowAveragesVector = self.rowSumsVector / len(self.rowSumsVector)
            self.columnAveragesVector = self.columnSumsVector / len(self.columnSumsVector)
            self.columnAveragesMatrix = np.outer(np.ones(len(self.rowSumsVector)), self.columnAveragesVector)
            self.rowAveragesMatrix = np.outer(self.rowAveragesVector, np.ones(len(self.columnSumsVector)))
            self.processedDataRepresentation = self.columnAveragesMatrix + self.rowAveragesMatrix
        if self.typeOfSystem == 2:
            self.coreDataRepresentation = self.trainsetDataRepresentation[1:, 1:]
            self.numberOfCoreDataRepresentationComponents = min(self.coreDataRepresentation.shape)
            self.U, self.S, self.VT = np.linalg.svd(self.coreDataRepresentation)
            k = self.S.shape[0]
            k = min(k, self.numberOfDimensionsToBeLeft)

            self.Uk = self.U[:, :k]
            self.VTk = self.VT[:k, :]

            self.Sk = np.diag(self.S[:k])

            self.processedDataRepresentation_ = np.dot(self.Sk, self.VTk)
            self.processedDataRepresentation = np.dot(self.Uk, self.processedDataRepresentation_)

        if self.typeOfSystem == 4:
            self.coreDataRepresentation = self.trainsetDataRepresentation[1:, 1:]
            self.numberOfCoreDataRepresentationComponents = min(self.coreDataRepresentation.shape)
            self.rowSumsVector = self.trainsetDataRepresentation[1:, 0]
            self.columnSumsVector = self.trainsetDataRepresentation[0, 1:]
            self.overallSumScalar = self.trainsetDataRepresentation[0, 0]
            self.rowAveragesVector = self.rowSumsVector / len(self.rowSumsVector)
            self.columnAveragesVector = self.columnSumsVector / len(self.columnSumsVector)
            self.overallAverageScalar = self.overallSumScalar / (len(self.rowSumsVector) * len(self.columnSumsVector))
            self.columnAveragesMatrix = np.outer(np.ones(len(self.rowSumsVector)), self.columnAveragesVector)
            self.rowAveragesMatrix = np.outer(self.rowAveragesVector, np.ones(len(self.columnSumsVector)))
            self.averagesBasedDataRepresentation = (
                                                       self.columnAveragesMatrix + self.rowAveragesMatrix) - self.overallAverageScalar

            self.coreDataRepresentation = self.trainsetDataRepresentation[1:, 1:] - self.averagesBasedDataRepresentation

            self.U, self.S, self.VT = np.linalg.svd(self.coreDataRepresentation)
            k = self.S.shape[0]
            k = min(k, self.numberOfDimensionsToBeLeft)

            self.Uk = self.U[:, :k]
            self.VTk = self.VT[:k, :]

            self.Sk = np.diag(self.S[:k])

            self.processedDataRepresentation_ = np.dot(self.Sk, self.VTk)
            self.processedDataRepresentation = np.dot(self.Uk, self.processedDataRepresentation_)

            self.processedDataRepresentation = self.processedDataRepresentation + self.averagesBasedDataRepresentation

        if self.typeOfSystem == 5:
            self.coreDataRepresentation = self.trainsetDataRepresentation[1:, 1:]
            self.numberOfCoreDataRepresentationComponents = min(self.coreDataRepresentation.shape)
            self.rowSumsVector = self.trainsetDataRepresentation[1:, 0]
            self.rowNonZeroElements = np.zeros(len(self.rowSumsVector))
            for i in range(len(self.rowNonZeroElements)):
                self.rowNonZeroElements[i] = np.count_nonzero(self.coreDataRepresentation[i, :])

            self.columnSumsVector = self.trainsetDataRepresentation[0, 1:]
            self.columnNonZeroElements = np.zeros(len(self.columnSumsVector))
            for i in range(len(self.columnNonZeroElements)):
                self.columnNonZeroElements[i] = np.count_nonzero(self.coreDataRepresentation[:, i])

            self.nonzeroElements = np.count_nonzero(self.coreDataRepresentation[:, :])
            self.overallSumScalar = self.trainsetDataRepresentation[0, 0]

            self.rowAveragesVector = np.zeros(len(self.rowSumsVector))
            for i in range(len(self.rowSumsVector)):
                self.rowAveragesVector[i] = self.rowSumsVector[i] / (self.rowNonZeroElements[i])

            self.columnAveragesVector = np.zeros(len(self.columnSumsVector))
            for i in range(len(self.columnSumsVector)):
                self.columnAveragesVector[i] = self.columnSumsVector[i] / (self.columnNonZeroElements[i])

            self.overallAverageScalar = self.overallSumScalar / (self.nonzeroElements)

            self.columnAveragesMatrix = np.outer(np.ones(len(self.rowSumsVector)), self.columnAveragesVector)
            self.rowAveragesMatrix = np.outer(self.rowAveragesVector, np.ones(len(self.columnSumsVector)))
            self.averagesBasedDataRepresentation = (
                                                       self.columnAveragesMatrix + self.rowAveragesMatrix) - self.overallAverageScalar

            # TYMCZASOWO:
            for i in range(len(self.rowSumsVector)):
                for j in range(len(self.columnSumsVector)):
                    if self.trainsetDataRepresentation[i + 1, j + 1] != 0:
                        self.coreDataRepresentation[i, j] = self.trainsetDataRepresentation[i + 1, j + 1] - \
                                                            self.rowAveragesVector[i] - self.columnAveragesVector[
                                                                j] + self.overallAverageScalar

            self.U, self.S, self.VT = np.linalg.svd(self.coreDataRepresentation)
            k = self.S.shape[0]
            k = min(k, self.numberOfDimensionsToBeLeft)

            self.Uk = self.U[:, :k]
            self.VTk = self.VT[:k, :]

            self.Sk = np.diag(self.S[:k])

            self.processedDataRepresentation_ = np.dot(self.Sk, self.VTk)
            self.processedDataRepresentation = np.dot(self.Uk, self.processedDataRepresentation_)

            self.processedDataRepresentation = self.processedDataRepresentation + self.averagesBasedDataRepresentation

        if self.typeOfSystem == 6:
            self.coreDataRepresentation = self.trainsetDataRepresentation[1:, 1:]
            self.numberOfCoreDataRepresentationComponents = min(self.coreDataRepresentation.shape)
            self.rowSumsVector = self.trainsetDataRepresentation[1:, 0]
            self.rowNonZeroElements = np.zeros(len(self.rowSumsVector))
            for i in range(len(self.rowNonZeroElements)):
                self.rowNonZeroElements[i] = np.count_nonzero(self.coreDataRepresentation[i, :])

            self.columnSumsVector = self.trainsetDataRepresentation[0, 1:]
            self.columnNonZeroElements = np.zeros(len(self.columnSumsVector))
            for i in range(len(self.columnNonZeroElements)):
                self.columnNonZeroElements[i] = np.count_nonzero(self.coreDataRepresentation[:, i])

            self.nonzeroElements = np.count_nonzero(self.coreDataRepresentation[:, :])
            self.overallSumScalar = self.trainsetDataRepresentation[0, 0]

            self.rowAveragesVector = np.zeros(len(self.rowSumsVector))
            for i in range(len(self.rowSumsVector)):
                self.rowAveragesVector[i] = self.rowSumsVector[i] / (self.rowNonZeroElements[i])

            self.columnAveragesVector = np.zeros(len(self.columnSumsVector))
            for i in range(len(self.columnSumsVector)):
                self.columnAveragesVector[i] = self.columnSumsVector[i] / (self.columnNonZeroElements[i])

            self.overallAverageScalar = self.overallSumScalar / (self.nonzeroElements)

            self.columnAveragesMatrix = np.outer(np.ones(len(self.rowSumsVector)), self.columnAveragesVector)
            self.rowAveragesMatrix = np.outer(self.rowAveragesVector, np.ones(len(self.columnSumsVector)))
            self.averagesBasedDataRepresentation = (
                                                       self.columnAveragesMatrix + self.rowAveragesMatrix) - self.overallAverageScalar
            for i in range(len(self.rowSumsVector)):
                for j in range(len(self.columnSumsVector)):
                    if self.trainsetDataRepresentation[i + 1, j + 1] != 0:
                        self.coreDataRepresentation[i, j] = self.trainsetDataRepresentation[i + 1, j + 1] - \
                                                            self.rowAveragesVector[i] - self.columnAveragesVector[
                                                                j] + self.overallAverageScalar

            self.U, self.S, self.VT = np.linalg.svd(self.coreDataRepresentation)
            k = self.S.shape[0]
            k = min(k, self.numberOfDimensionsToBeLeft)
            self.Uk = self.U[:, :k]
            self.VTk = self.VT[:k, :]
            self.Sk = np.diag(self.S[:k])
            self.processedDataRepresentation_ = np.dot(self.Sk, self.VTk)
            self.processedDataRepresentation = np.dot(self.Uk, self.processedDataRepresentation_)

            self.processedDataRepresentation = self.averagesBasedDataRepresentation

        self.inputDataProcessed = True

    def getQueryFloatResult(self, queryTuple):
        # przypadek systemu najgorszego z mozliwych:
        if self.typeOfSystem == 0:
            #            #tu widac korzysc z tego, ze typ queryTuple jest "hashable":
            queryResult = random.random()

        # przypadek idealnego systemu:
        if self.typeOfSystem == 1:

            queryResult = self.sciagaDlaWszechwiedzacego[queryTuple]

        if (self.typeOfSystem == 2) or (self.typeOfSystem == 3) or (self.typeOfSystem == 4) or (
                    self.typeOfSystem == 5) or (self.typeOfSystem == 6) or self.typeOfSystem == 7:
            if (queryTuple[0] in self.dids1.listaSlownikowIndeksowania[1].keys()) and (
                        queryTuple[1] in self.dids1.listaSlownikowIndeksowania[2].keys()):
                queryResult = self.processedDataRepresentation[
                    self.dids1.listaSlownikowIndeksowania[1][queryTuple[0]] - 1,
                    self.dids1.listaSlownikowIndeksowania[2][queryTuple[1]] - 1]
            else:
                queryResult = 0
        return queryResult

    def getMultiQueryFloatResults(self, queryTuplesList):
        if not (self.inputDataProcessed):
            self.processInputArray()
        multiQueryFloatResults = {}
        for tempQueryTuple in queryTuplesList:
            multiQueryFloatResults[tempQueryTuple] = self.getQueryFloatResult(tempQueryTuple)
        return multiQueryFloatResults

    def spoilResults(self, sciaga):
        self.sciagaDlaWszechwiedzacego = sciaga


def getMultiQueryBinaryResults(recSys, tso, numberOfThresholdSteps):
    multiQueryFloatResults = recSys.getMultiQueryFloatResults(tso.queryTuplesList)
    minThreshold = min(list(multiQueryFloatResults.values()))
    maxThreshold = max(list(multiQueryFloatResults.values()))

    targetPositive2AllRatioStep = (maxThreshold - minThreshold) / (numberOfThresholdSteps + 2)
    listOfmultiQueryBinaryResults = []
    for intermediateTargetPositive2AllRatioStepNumber in range(numberOfThresholdSteps):
        threshold = ((intermediateTargetPositive2AllRatioStepNumber + 1) * targetPositive2AllRatioStep) + minThreshold
        multiQueryBinaryResults = []
        multiQueryFloatResults_ = []
        numberOfPositiveBinResults = 0
        for tempQueryTuple in tso.queryTuplesList:
            tempQueryFloatResult = multiQueryFloatResults[tempQueryTuple]

            tempQueryBinResult = -1
            if tempQueryFloatResult >= threshold:
                tempQueryBinResult = 1
                numberOfPositiveBinResults += 1

            multiQueryBinaryResults.append(tuple([tempQueryBinResult]) + tuple(tempQueryTuple))
            multiQueryFloatResults_.append(tuple([tempQueryFloatResult]) + tuple(tempQueryTuple))
        if numberOfPositiveBinResults > 0:
            listOfmultiQueryBinaryResults.append(multiQueryBinaryResults)
        else:
            print('cosik nie teges....')

    return listOfmultiQueryBinaryResults, multiQueryFloatResults


def getPrecisionVsRecallPointsForASingleCurve(multiQueryBinaryResults):
    recallValues = []
    precisionValues = []
    for currentPvsRCurvePoint in range(numberOfPvsRCurvePoints):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for currentMultiQueryResultTupleNumber in range(len(multiQueryBinaryResults[currentPvsRCurvePoint])):
            if ((multiQueryBinaryResults[currentPvsRCurvePoint][currentMultiQueryResultTupleNumber][0] == 1) and (
                        testSet[currentMultiQueryResultTupleNumber][0] == 1)):
                TP += 1
            if ((multiQueryBinaryResults[currentPvsRCurvePoint][currentMultiQueryResultTupleNumber][0] == -1) and (
                        testSet[currentMultiQueryResultTupleNumber][0] == -1)):
                TN += 1
            if ((multiQueryBinaryResults[currentPvsRCurvePoint][currentMultiQueryResultTupleNumber][0] == 1) and (
                        testSet[currentMultiQueryResultTupleNumber][0] == -1)):
                FP += 1
            if ((multiQueryBinaryResults[currentPvsRCurvePoint][currentMultiQueryResultTupleNumber][0] == -1) and (
                        testSet[currentMultiQueryResultTupleNumber][0] == 1)):
                FN += 1
        if TP + FP == 0:
            print("(TP+FP)==0")
            print(TP)
            print(FP)
        if TP + FN == 0:
            print("(TP+FN)==0")
            print(TP)
            print(FN)
        precisionValues.append(float(TP) / (TP + FP))
        recallValues.append(float(TP) / (TP + FN))
    return precisionValues, recallValues


def altGetPrecisionVsRecallPointsForASingleCurve(multiqueryFloatResults, testSet, numberOfPointsPerCurve=0):
    multiqueryFloatResultsAsList = []
    altPrecisionPoints = []
    altRecallPoints = []

    for tempMultiqueryFloatResult in multiqueryFloatResults:
        tempMultiqueryFloatResultsAsListElement = [multiqueryFloatResults[tempMultiqueryFloatResult],
                                                   tempMultiqueryFloatResult]
        multiqueryFloatResultsAsList.append(tempMultiqueryFloatResultsAsListElement)
    multiqueryFloatResultsAsListSorted = sorted(multiqueryFloatResultsAsList, key=lambda x: x[0], reverse=True)

    testSetSelected = set()
    for tempTestSetSortedElement in testSet:
        if tempTestSetSortedElement[0] == 1:
            testSetSelected.add((tempTestSetSortedElement[1:]))
    totalNumberOfHits = len(testSetSelected)
    if numberOfPointsPerCurve == 0:
        numberOfPointsPerCurve = totalNumberOfHits
    curveGrain = int(float(totalNumberOfHits) / numberOfPointsPerCurve)
    tempNumberOfHits = 0
    tempNumberOfShots = 0
    curveGrainCounter = 0
    for tempMultiqueryFloatResultsAsListSortedElement in multiqueryFloatResultsAsListSorted:

        tempNumberOfShots += 1
        if tempMultiqueryFloatResultsAsListSortedElement[1] in testSetSelected:
            tempNumberOfHits += 1

            tempAltPrecisionPointsValue = float(tempNumberOfHits) / tempNumberOfShots

            curveGrainCounter = (curveGrainCounter + 1) % curveGrain
            if curveGrainCounter == 0:
                altPrecisionPoints.append(tempAltPrecisionPointsValue)

                altRecallPoints.append(float(tempNumberOfHits) / totalNumberOfHits)
    return altPrecisionPoints, altRecallPoints


def getTPVsPPointsForASingleCurve(multiQueryBinaryResults):
    TPValues = []
    PValues = []
    numberOfCurvePoints = len(multiQueryBinaryResults)
    for curvePoint in range(numberOfCurvePoints):
        TP = 0
        P = 0
        for currentMultiQueryResultTupleNumber in range(len(multiQueryBinaryResults[curvePoint])):
            if ((multiQueryBinaryResults[curvePoint][currentMultiQueryResultTupleNumber][0] == 1) and (
                        testSet[currentMultiQueryResultTupleNumber][0] == 1)):
                TP += 1
            if (multiQueryBinaryResults[curvePoint][currentMultiQueryResultTupleNumber][0] == 1):
                P += 1
        TPValues.append(TP)
        PValues.append(P)
    return TPValues, PValues


def makeSpectrumFigure(spctr, numberOfComponents, fileName):
    if numberOfComponents == 0:
        numberOfComponents = len(list(spctr[0][0]))
    plt.ioff()
    plt.figure(figsize=(20, 10), linewidth=0.1)
    for row in range(len(spctr)):
        for column in range(len(spctr[row])):
            trainingRatio = round((column + 1) * trainingRatioStep, 10)
            plt.subplot(len(spctr), len(spctr[0]), len(spctr[0]) * row + column + 1)
            plt.plot(list(spctr[row][column])[:numberOfComponents])
            axes = plt.gca()
            axes.spines['left'].set_linewidth(0.2)
            axes.spines['right'].set_linewidth(0.2)
            axes.spines['bottom'].set_linewidth(0.2)
            axes.spines['top'].set_linewidth(0.2)
            title = "randomTrainAndTestSetSplitCase#" + str(row) + ", trainingRatio=" + str(round(trainingRatio, 2))
            plt.title(title, fontsize=6, y=1.03)
            leg.get_frame().set_linewidth(0.05)
    plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 6})
    plt.savefig('outputs/' + fileName + ".png", format="png", dpi=100)
    plt.clf()


def makeCurveFigure(horizontalValues, verticalValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps,
                    numberOfRecSystems, recSystemsLabels, horizontalAxisLabel, verticalAxisLabel, fileName,
                    legendLocation):
    plt.ioff()
    plt.figure(figsize=(20, 10), linewidth=0.1)
    for randomTrainAndTestSetSplitCaseNumber in range(numberOfRandomTrainAndTestSetSplitCases):
        for currentTRStepNumber in range(numberOfTRSteps):
            trainingRatio = round((currentTRStepNumber + 1) * trainingRatioStep, 10)
            plt.subplot(numberOfRandomTrainAndTestSetSplitCases + 0, numberOfTRSteps + 0,
                        currentTRStepNumber + 1 + ((numberOfTRSteps + 0) * (randomTrainAndTestSetSplitCaseNumber + 0)))
            for systemNumber in range(numberOfRecSystems):

                plt.plot(horizontalValues[randomTrainAndTestSetSplitCaseNumber][currentTRStepNumber][systemNumber],
                         verticalValues[randomTrainAndTestSetSplitCaseNumber][currentTRStepNumber][systemNumber],
                         marker='o', markersize=1, label=recSystemsLabels[systemNumber])
                axes = plt.gca()
                axes.spines['left'].set_linewidth(0.2)
                axes.spines['right'].set_linewidth(0.2)
                axes.spines['bottom'].set_linewidth(0.2)
                axes.spines['top'].set_linewidth(0.2)
                #                axes.set_xlim([0,1])
                #                axes.set_ylim([0,1])
                title = "randomTrainAndTestSetSplitCase#" + str(
                    randomTrainAndTestSetSplitCaseNumber) + ", trainingRatio=" + str(round(trainingRatio, 2))
                #            plt.title(title)
                plt.title(title, fontsize=6, y=1.03)
                leg = plt.legend(loc=legendLocation, fontsize=6)
                leg.get_frame().set_linewidth(0.05)
                #            plt.xlabel(horizontalAxisLabel)
                #            plt.ylabel(verticalAxisLabel)
                plt.xlabel(horizontalAxisLabel, fontsize=6)
                plt.ylabel(verticalAxisLabel, fontsize=6)

    plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 6})

    plt.savefig(fileName + ".png", format="png", dpi=400)
    plt.clf()


if __name__ == '__main__':

    wspolczynnikRedukcjiWielkosciZbioruDanych = 0.1

    SubUDataSet = getSubUDataSet(wspolczynnikRedukcjiWielkosciZbioruDanych, 1)
    inTuples = convertSubUDataSetToInTuplesList(SubUDataSet, 4)

    dataSetSize = len(inTuples)

    precisionValues = []

    altPrecisionValues = []
    altRecallValues = []

    TPValues = []
    recallValues = []
    PValues = []
    TrainSets = []
    TestSets = []
    TestResults = []
    TestResults2 = []

    numberOfRecSystems = 8  # 7

    numberOfTRSteps = 4
    trainingRatioStep = 1.0 / (numberOfTRSteps + 1)

    spectrum = []
    sortedRowSums = []
    sortedColumnSums = []

    numberOfRandomTrainAndTestSetSplitCases = 3

    random.shuffle(inTuples)
    random.shuffle(inTuples)
    random.shuffle(inTuples)
    random.shuffle(inTuples)

    fileForInTuples = open("outputs/fileForInTuples", 'w')
    fileForInTuples.write(str(inTuples))
    fileForInTuples.close()

    for randomTrainAndTestSetSplitCaseNumber in range(numberOfRandomTrainAndTestSetSplitCases):
        random.shuffle(inTuples)
        spectrum.append([])
        sortedRowSums.append([])
        sortedColumnSums.append([])
        precisionValues.append([])
        TPValues.append([])
        recallValues.append([])

        altPrecisionValues.append([])
        altRecallValues.append([])

        PValues.append([])
        TrainSets.append([])
        TestSets.append([])
        TestResults.append([])
        TestResults2.append([])

        for currentTRStepNumber in range(numberOfTRSteps):
            random.shuffle(inTuples)
            trainingRatio = round((currentTRStepNumber + 1) * trainingRatioStep, 10)
            print('currentTRStepNumber: {}'.format(currentTRStepNumber))
            precisionValues[-1].append([])
            TPValues[-1].append([])
            recallValues[-1].append([])

            altPrecisionValues[-1].append([])
            altRecallValues[-1].append([])

            PValues[-1].append([])
            TestResults[-1].append([])
            TestResults2[-1].append([])

            trainSetSize = int(trainingRatio * dataSetSize)
            print('trainSetSize: {}'.format(trainSetSize))
            numberOfPositivesInTestset = 0
            numberOfNegativesInTestset = 0
            while not numberOfPositivesInTestset or not numberOfNegativesInTestset:
                random.shuffle(inTuples)
                testSet = inTuples[trainSetSize:]
                numberOfPositivesInTestset = 0
                numberOfNegativesInTestset = 0

                for tempTestsetTuple in testSet:
                    if tempTestsetTuple[0] == 1:
                        numberOfPositivesInTestset += 1
                    if tempTestsetTuple[0] == -1:
                        numberOfNegativesInTestset += 1
            trainSet = inTuples[:trainSetSize]
            TestSets[-1].append(testSet)
            TrainSets[-1].append(trainSet)

            dids1 = DimensionsIndexingDictionaries(trainSet)
            dids2 = DimensionsIndexingDictionaries(testSet)

            abdr1 = ArrayBasedDataRepresentation(dids1, trainSet)
            spectrum[-1].append(getMatrixSpectrum(abdr1.trainsetDataRepresentation))

            abdr2 = ArrayBasedDataRepresentation(dids2, testSet)

            tso = TestSetObject(testSet)

            recSystems = []

            rs = RecSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 0)
            recSystems.append(rs)

            rs = RecSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 1)
            rs.spoilResults(tso.sciagaDlaWszechwiedzacego)
            recSystems.append(rs)

            rs = RecSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 2)
            rs.numberOfDimensionsToBeLeft = 10
            recSystems.append(rs)

            rs = RecSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 3)
            recSystems.append(rs)

            rs = RecSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 4)
            rs.numberOfDimensionsToBeLeft = 10
            recSystems.append(rs)

            rs = RecSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 5)
            rs.numberOfDimensionsToBeLeft = 10
            recSystems.append(rs)

            rs = RecSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 6)
            rs.numberOfDimensionsToBeLeft = 10
            recSystems.append(rs)

            rs = RecSystem(dids1, copy.deepcopy(abdr1.trainsetDataRepresentation), 7)
            rs.numberOfDimensionsToBeLeft = 10
            recSystems.append(rs)

            numberOfPvsRCurvePoints = 20
            for recSysNumber in range(numberOfRecSystems):
                currentRecSystem = recSystems[recSysNumber]

                multiQueryBinaryResults, multiqueryFloatResults_ = getMultiQueryBinaryResults(currentRecSystem, tso,
                                                                                              numberOfPvsRCurvePoints)
                precisionPoints, recallPoints = getPrecisionVsRecallPointsForASingleCurve(multiQueryBinaryResults)

                altPrecisionPoints, altRecallPoints = altGetPrecisionVsRecallPointsForASingleCurve(
                    multiqueryFloatResults_, testSet, 20)

                TPPoints, PPoints = getTPVsPPointsForASingleCurve(multiQueryBinaryResults)

                precisionValues[-1][-1].append(precisionPoints)
                TPValues[-1][-1].append(TPPoints)
                recallValues[-1][-1].append(recallPoints)

                altPrecisionValues[-1][-1].append(altPrecisionPoints)
                altRecallValues[-1][-1].append(altRecallPoints)

                PValues[-1][-1].append(PPoints)
                TestResults[-1][-1].append(multiQueryBinaryResults)
                TestResults2[-1][-1].append(multiqueryFloatResults_)

    recSystemsLabels = ['random', 'ideal', 'SVD-based', 'classical-averages-based', 'overallCentring-based_MPCA',
                        'statisticalCentring-based_MPCA with decentring', 'statistical-avaraged-based', 'my-rec-system']

    plt.ioff()
    plt.figure(figsize=(20, 10), linewidth=0.1)

    for randomTrainAndTestSetSplitCaseNumber in range(numberOfRandomTrainAndTestSetSplitCases):
        for currentTRStepNumber in range(numberOfTRSteps):
            trainingRatio = round((currentTRStepNumber + 1) * trainingRatioStep, 10)
            plt.subplot(numberOfRandomTrainAndTestSetSplitCases + 0, numberOfTRSteps + 0,
                        currentTRStepNumber + 1 + ((numberOfTRSteps + 0) * (randomTrainAndTestSetSplitCaseNumber + 0)))
            for systemNumber in range(numberOfRecSystems):
                plt.plot(recallValues[randomTrainAndTestSetSplitCaseNumber][currentTRStepNumber][systemNumber],
                         precisionValues[randomTrainAndTestSetSplitCaseNumber][currentTRStepNumber][systemNumber],
                         marker='o', markersize=3, label=recSystemsLabels[systemNumber])
                axes = plt.gca()
                axes.spines['left'].set_linewidth(0.2)
                axes.spines['right'].set_linewidth(0.2)
                axes.spines['bottom'].set_linewidth(0.2)
                axes.spines['top'].set_linewidth(0.2)
                axes.set_xlim([0, 1])
                axes.set_ylim([0, 1])
                title = "randomTrainAndTestSetSplitCase#" + str(
                    randomTrainAndTestSetSplitCaseNumber) + ", trainingRatio=" + str(round(trainingRatio, 2))
                plt.title(title, fontsize=6, y=1.03)
                leg = plt.legend(loc=3, fontsize=6)
                leg.get_frame().set_linewidth(0.05)
                plt.xlabel("Recall", fontsize=6)
                plt.ylabel("Precision", fontsize=6)

    plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 6})
    plt.savefig("outputs/wykresPvsR1.png", format="png", dpi=100)
    plt.clf()

    makeCurveFigure(PValues, TPValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps, numberOfRecSystems,
                    recSystemsLabels, 'number of Positives', 'number of True Positives',
                    'outputs/PositivesVsTruePositives', 0)
    PvsTPFigureData = (
        PValues, TPValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps, numberOfRecSystems,
        recSystemsLabels,
        'number of Positives', 'number of True Positives', 'PositivesVsTruePositives', 0)

    makeCurveFigure(recallValues, precisionValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps,
                    numberOfRecSystems, recSystemsLabels, 'Recall', 'Precision', 'outputs/PrecisionVsRecall_2', 0)

    PvsRFigureData = (
        recallValues, precisionValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps, numberOfRecSystems,
        recSystemsLabels, 'Recall', 'Precision', 'PrecisionVsRecall_3', 0)

    makeCurveFigure(altRecallValues, altPrecisionValues, numberOfRandomTrainAndTestSetSplitCases, numberOfTRSteps,
                    numberOfRecSystems, recSystemsLabels, 'Recall', 'Precision', 'outputs/PrecisionVsRecall_4', 0)

    # legend location values:
    # best -- 0
    # upper right -- 1
    # upper left -- 2
    # lower left -- 3
    # lower right -- 4
    # right -- 5
    # center left -- 6
    # center right -- 7
    # lower center -- 8
    # upper center -- 9
    # center -- 10

    makeSpectrumFigure(spectrum, 0, 'full_trainsets_spectra')
    makeSpectrumFigure(spectrum, 5, 'principal_components_of_trainsets_spectra')

"""
In this case we only use the training data from the same district of the test data
"""
# from parserJSON import *
from data_preprocessing import parserImproved
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold  # import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import time
from sklearn.model_selection import train_test_split
import pickle
from collections import defaultdict

output = open('smNeighbourhoodPrice.pkl', 'rb'); smNeighbourhoodPrice = pickle.load(output); output.close()
output = open('smDistrictPrice.pkl', 'rb'); smDistrictPrice = pickle.load(output); output.close()
output = open('dataByNeighborhood.pkl', 'rb'); dataByNeighborhood = pickle.load(output); output.close()
output = open('dataByDistrict.pkl', 'rb'); dataByDistrict = pickle.load(output); output.close()
output = open('neighbourhoodDistrict.pkl', 'rb'); neighbourhoodDistrict = pickle.load(output); output.close()
# districtNames = dataByDistrict.keys()

# classify each neighbourhood by district
districtNeighborhood = defaultdict(list)
for neigh in neighbourhoodDistrict:
    districtNeighborhood[neighbourhoodDistrict[neigh]].append(neigh)

# extract the average m^2 price of each neighborhood and district
ngPrices = []; dtPrices = []; districtNames = []; neighbourNames = []
for dt in districtNeighborhood:
    districtNames.append(dt)
    dtPrices.append(smDistrictPrice[dt])

    for neigh in districtNeighborhood[dt]:
        neighbourNames.append(neigh)
        ngPrices.append(smNeighbourhoodPrice[neigh])

# dataHeaders = ["Floor", "PropertyType", "m2", "exterior", "bedrooms", "bathrooms", "district", "status", "newDevelopment", "hasLift", "parking", "areaprice", "finalPrice"]
# dataHeaders = ["Floor", "hasLift", "district", "neighborhood", "detailedTypeTypology", "detailedTypeSubTypology", "newDevelopment", "priceByArea", "rooms", "size", "status", "exterior", "bathrooms", "parkingSpaceHasParkingSpace", "parkingSpaceIsParkingSpaceIncludedInPrice"]
dataHeaders = ["Floor", "hasLift", "detailedTypeTypology", "detailedTypeSubTypology", "newDevelopment", "rooms", "size", "status", "exterior", "bathrooms", "parkingSpaceHasParkingSpace", "parkingSpaceIsParkingSpaceIncludedInPrice", "districtPrice_m2"]

#################### CLASSIFIERS ########################
# title for the plots
names = ('Gradien boosting',
         'Extra trees',
         'Random forest',
         'Elastic net',
         'Lasso reg',
         'Ridge reg',
         'Bayesian ridge',
         'SVR linear 1C',
         'SVR linear 100C',
         'SVR rbf 1C',
         'SVR rbf 100C',
         'SVR poly 1C',
         'MLPReg lbfgs 15ly',
         'MLPReg sgd 15ly',
         'MLPReg lbfgs 20ly',
         'MLPReg sgd 20ly',
         'MLPReg lbfgs 100ly',
         'MLPReg sgd 100ly',
         'KNNRegressor 3NN')

classifiers = (GradientBoostingRegressor(),
               ExtraTreesRegressor(),
               RandomForestRegressor(),
               linear_model.ElasticNet(alpha=0.1),
               linear_model.Lasso(alpha = 0.1),
               linear_model.Ridge(alpha=.5),
               linear_model.BayesianRidge(),
               SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
               SVR(C=100.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
               SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
               SVR(C=100.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
               SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
               MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1),
               MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1),
               MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1),
               MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1),
               MLPRegressor(solver='lbfgs'),
               MLPRegressor(solver='sgd'),
               KNeighborsRegressor(n_neighbors=3, weights='distance'))

#########################################################
bestMeanValues = []
bestMaxValues = []
bestMinValues = []
bestNames = []
for dt in districtNames:
    X = []; Y = []
    dtPrice = smDistrictPrice[dt]
    for inst in dataByDistrict[dt]:
        # inst[0].append(dtPrice)
        X.append(inst[0])
        Y.append(inst[1])

    # X, Y = parserfromtxt("./data/definitiveInstanceFile.txt")
    rang = 'pos' #'posNeg'
    preProcessedX, preProcessedY, newDataHeaders = parserImproved(X, Y, rang, dataHeaders, scaling=True, normalizing=True, unicolumn=False)
    # instancestotxt(preProcessedX, preProcessedY)

    # feature selection
    print "\nRESUTLTS WITH FEATURE SELECTION"

    npPreprocessedX = np.array(preProcessedX)
    currentData = npPreprocessedX
    originalResults = np.zeros_like(classifiers)
    removedFeatures = []
    bestRoundResultsIndex = -1
    first = True
    bestRoundResults = 0
    bestAllResults = 0
    innerFeature = 0
    bestAllResultsSelection = range(len(npPreprocessedX))
    endBucle = False
    while len(currentData[0]) > 1 or endBucle:
        currentRoundResults = np.zeros_like(classifiers)
        currentData = npPreprocessedX
        currentRoundRemovedFeatures = []
        for f in range(len(removedFeatures)):
            currentRoundRemovedFeatures.append(removedFeatures[f])
        if not first:
            currentData = np.delete(npPreprocessedX, innerFeature, 1)
            for rFeature in range(len(currentRoundRemovedFeatures)):
                if innerFeature < currentRoundRemovedFeatures[rFeature]:
                    currentRoundRemovedFeatures[rFeature] -= 1
            for feature in currentRoundRemovedFeatures:
                currentData = np.delete(currentData, feature, 1)
                for rFeature in range(len(currentRoundRemovedFeatures)):
                    if feature < currentRoundRemovedFeatures[rFeature]:
                        currentRoundRemovedFeatures[rFeature] -= 1

        for reg, name, regCount in zip(classifiers, names, range(len(classifiers))):
            if regCount > 0:  # only for debugging purposes
                break
            cv_results = cross_val_score(reg, currentData, preProcessedY, cv=2)
            cv_results_mean = round(np.mean(cv_results), 3)

            if first:
                originalResults[regCount] = cv_results_mean
                print "originalAccuracy: {}".format(np.max(originalResults))
            else:
                currentRoundResults[regCount] = cv_results_mean

        if np.max(currentRoundResults) > np.max(bestRoundResults) and not first:
            bestRoundResults = currentRoundResults
            bestRoundResultsIndex = innerFeature

            bestRoundFeatureSelection = []
            for i in range(len(npPreprocessedX[0])):
                if i not in removedFeatures:
                    bestRoundFeatureSelection.append(i)

        if first:
            first = False
        else:
            innerFeature += 1
            while innerFeature in removedFeatures:
                innerFeature += 1

        if innerFeature >= len(preProcessedX[0]) - 1:
            removedFeatures.append(bestRoundResultsIndex)

            if np.max(bestRoundResults) < 0.5 * np.max(originalResults):
                endBucle = True

            print "\nbestRoundFeatureSelection: {}".format(bestRoundFeatureSelection)
            print "bestRoundResults: {}".format(np.max(bestRoundResults))

            if np.max(bestRoundResults) >= np.max(bestAllResults):  # it is >= because if same accuracy is achieved with less features, this new simple subset is preferred
                bestAllResults = bestRoundResults
                bestAllResultsSelection = []
                for i in range(len(npPreprocessedX[0])):
                    if i not in removedFeatures:
                        bestAllResultsSelection.append(i)
            bestRoundResults = 0
            innerFeature = 0

    bestAllResultsSelectionNames = []
    for sf in range(len(bestAllResultsSelection)):
        bestAllResultsSelectionNames.append(newDataHeaders[sf])

    print "\nbestAllResultsSelection: {}".format(bestAllResultsSelection)
    print "\nbestAllResultsSelectionNames: {}".format(bestAllResultsSelectionNames)
    print "Best accuracy achieved: {}".format(np.max(bestAllResults))


    ### FINISH OF FEATURE SELECTION PART
    bestMeanValues.append(bestMeanAccuracy)
    bestMaxValues.append(bestMaxAccuracy)
    bestMinValues.append(bestMinAccuracy)
    bestNames.append([bestMeanAccuracyName, bestMaxAccuracyName, bestMinAccuracyName])
    print "mean:{} with {} max:{} with min:{} with {}".format(bestMeanAccuracy, bestMeanAccuracyName, bestMaxAccuracy, bestMaxAccuracyName, bestMinAccuracy, bestMinAccuracyName)

bestMeanAccuracyIndex = np.argmax(bestMeanValues)
bestMaxAccuracyIndex = np.argmax(bestMaxValues)
bestMaxAccuracyIndex = np.argmax(bestMinValues)
bestMeanAccuracy = bestMeanValues[bestMeanAccuracyIndex]
bestMaxAccuracy = bestMaxValues[bestMeanAccuracyIndex]
bestMinAccuracy = bestMinValues[bestMeanAccuracyIndex]
bestMeanAccuracyName = bestNames[bestMeanAccuracyIndex][0]
bestMaxAccuracyName = bestNames[bestMeanAccuracyIndex][1]
bestMinAccuracyName = bestNames[bestMeanAccuracyIndex][2]

print "BEST RESULTS: mean:{} with {} max:{} with {} min:{} with {}".format(bestMeanAccuracy, bestMeanAccuracyName, bestMaxAccuracy, bestMaxAccuracyName, bestMinAccuracy, bestMinAccuracyName)

pass


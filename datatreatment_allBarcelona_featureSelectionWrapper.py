"""
In this case we scale (min max) each batch of data by district and then join all data to fit the regressors and perform predictions
"""
# from parserJSON import *
from data_preprocessing import *
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
import time
from sklearn.model_selection import train_test_split
import pickle
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

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
ngPrices = [];
dtPrices = [];
districtNames = [];
neighbourNames = []
for dt in districtNeighborhood:
    districtNames.append(dt)
    dtPrices.append(smDistrictPrice[dt])

    for neigh in districtNeighborhood[dt]:
        neighbourNames.append(neigh)
        ngPrices.append(smNeighbourhoodPrice[neigh])
# dataHeaders = ["Floor", "PropertyType", "m2", "exterior", "bedrooms", "bathrooms", "district", "status", "newDevelopment", "hasLift", "parking", "areaprice", "finalPrice"]

# dataHeaders = ["Floor", "hasLift", "district", "neighborhood", "detailedTypeTypology", "detailedTypeSubTypology", "newDevelopment", "priceByArea", "rooms", "size", "status", "exterior", "bathrooms", "parkingSpaceHasParkingSpace", "parkingSpaceIsParkingSpaceIncludedInPrice"]
dataHeaders = ["Floor", "hasLift", "detailedTypeTypology", "detailedTypeSubTypology", "newDevelopment", "rooms", "size", "status", "exterior", "bathrooms", "parkingSpaceHasParkingSpace",
               "parkingSpaceIsParkingSpaceIncludedInPrice", "districtPrice_m2"]

#################### CLASSIFIERS ########################
# names of the regressors
names = ('Gradient boosting',
         'Extra trees',
         'Random forest',
         # 'Elastic net',
         # 'Lasso reg',
         # 'Ridge reg',
         # 'Bayesian ridge',
         'SVR linear',
         'MLPReg',
         'KNNRegressor')

# regressors sklearn models
classifiers = (GradientBoostingRegressor(),
               ExtraTreesRegressor(),
               RandomForestRegressor(),
               # linear_model.ElasticNet(),
               # linear_model.Lasso(),
               # linear_model.Ridge(),
               # linear_model.BayesianRidge(),
               SVR(),
               MLPRegressor(),
               KNeighborsRegressor())

# Param grids
param_grid_GradientBoosting = {"loss": ['ls', 'lad', 'huber', 'quantile'],
              "learning_rate": [0.1, 0.3],
              "n_estimators": [100, 150]}

param_grid_ExtraTrees = {"n_estimators": [5, 10, 20],
              "max_features": ['auto', 'sqrt', 'log2', None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "criterion": ["mse", "mae"]}

param_grid_RandomForest = {"max_depth": [3, None],
                          "max_features": ['auto', 'sqrt', 'log2', None],
                          "min_samples_split": [2, 3, 10],
                          "min_samples_leaf": [1, 3, 10],
                          "criterion": ["mse", "mae"]}

param_grid_ElasticNet = {"alpha": [0.1, 0.5, 1, 10, 50]}

param_grid_Lasso = {"alpha": [0.1, 0.5, 1, 10, 50]}

param_grid_Ridge = {"alpha": [0.1, 0.5, 1, 10, 50],
                    "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

param_grid_BayesianRidge = {"alpha_1": [0.5e-6, 1.e-6, 2.e-6],
                            "alpha_2": [0.5e-6, 1.e-6, 2.e-6],
                            "lambda_1": [0.5e-6, 1.e-6, 2.e-6],
                            "lambda_2": [0.5e-6, 1.e-6, 2.e-6]}

param_grid_SVR = {"C": [0.5, 1, 10],
                "epsilon": [0.05, 0.1, 1, 2],
                "kernel": ['rbf', 'linear', 'poly', 'sigmoid']}

param_grid_MLP = {'hidden_layer_sizes ': [(15,), (20,), (100,)],
    #"activation": ['identity', 'logistic', 'tanh', 'relu'],
    "solver": ['lbfgs', 'sgd', 'adam'],
    #"learning_rate": ['constant', 'invscaling', 'adaptive'],
    "batch_size": [1, 'auto']}  # ,
    # "momentum": [0.8, 0.9, 1.0]}

param_grid_KNN = {"n_neighbors": [1, 5, 10],
                  "weights": ['uniform', 'distance']}

param_grids = (param_grid_GradientBoosting,
               param_grid_ExtraTrees,
               param_grid_RandomForest,
               # param_grid_ElasticNet,
               # param_grid_Lasso,
               # param_grid_Ridge,
               # param_grid_BayesianRidge,
               param_grid_SVR,
               param_grid_MLP,
               param_grid_KNN)

#########################################################
data = []
labels = []
rang = 'pos'  # 'posNeg'
for dt in districtNames:
    X = [];
    Y = []
    dtPrice = smDistrictPrice[dt]
    for inst in dataByDistrict[dt]:
        inst[0].append(dtPrice)   # here we add the districtPrice_m2
        X.append(inst[0])
        Y.append(inst[1])

    X, Y = minMaxScaling(X, Y, rang, dataHeaders)  # scale all features but "districtPrice_m2"

    for instIndex in range(len(X)):
        data.append(X[instIndex])
        labels.append(Y[instIndex])

# HERE WE HAVE ALL BARCELONA DATA JOINED, NOW WE COULD PROCEED TO SCALE AND NORMALIZE ALL DATA TOGETHER, also districtPrice_m2
preProcessedX, preProcessedY, newDataHeaders = parserImproved(data, labels, rang, dataHeaders, scaling=True, normalizing=True, unicolumn=False)

# output = open('newDataHeaders.pkl', 'wb'); pickle.dump(newDataHeaders, output); output.close()

# feature selection, this part of code is commented in datatreatment_byDistricts_featureSelectionWrapper.py
print "\nRESULTS WITH FEATURE SELECTION"

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

    for reg, name, regCount, param_grid in zip(classifiers, names, range(len(classifiers)), param_grids):
        # if regCount > 0:  # only for debugging purposes
        #     break
        # gridReg = GridSearchCV(reg, param_grid)
        # cv_results = cross_val_score(gridReg, currentData, preProcessedY, cv=10)
        cv_results = cross_val_score(reg, currentData, preProcessedY, cv=4)
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
# output = open('bestAllResultsSelection.pkl', 'wb'); pickle.dump(bestAllResultsSelection, output); output.close()

print "\nbestAllResultsSelectionNames: {}".format(bestAllResultsSelectionNames)
print "Best accuracy achieved: {}".format(np.max(bestAllResults))

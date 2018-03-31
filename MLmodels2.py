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

from sklearn.metrics import make_scorer

output = open('newDataHeaders.pkl', 'rb'); selectedDataHeaders = pickle.load(output); output.close()
output = open('bestAllResultsSelection.pkl', 'rb'); bestAllResultsSelection = pickle.load(output); output.close()
output = open('smNeighbourhoodPrice.pkl', 'rb'); smNeighbourhoodPrice = pickle.load(output); output.close()
output = open('smDistrictPrice.pkl', 'rb'); smDistrictPrice = pickle.load(output); output.close()
output = open('dataByNeighborhood.pkl', 'rb'); dataByNeighborhood = pickle.load(output); output.close()
output = open('dataByDistrict.pkl', 'rb'); dataByDistrict = pickle.load(output); output.close()
output = open('neighbourhoodDistrict.pkl', 'rb'); neighbourhoodDistrict = pickle.load(output); output.close()

def custom_loss_func(data, predictions):
    diff = np.abs(data - predictions)
    return diff

scoring = {'loss_fnc': make_scorer(custom_loss_func)}

# classify each neighbourhood by district
districtNeighborhood = defaultdict(list)
for neigh in neighbourhoodDistrict:
    districtNeighborhood[neighbourhoodDistrict[neigh]].append(neigh)

# DEALING WITH MANUAL INSTANCES, any number of manual instances could be introduced next
initialPib = 1.0
currentPib = 1.0

realPrice1 = 440000
instance1 = ["Floor", 4.0,  # float
             "hasLift", True,  # boolean
             "detailedTypeTypology", 'flat',  # countryHouse, duplex, independantHouse, penthouse, ?
             "detailedTypeSubTypology", 'penthouse',  # countryHouse, duplex, independantHouse, penthouse, terracedHouse, ?
             "newDevelopment", True,  # boolean
             "rooms", 4,  # integer
             "size", 95,  # integer (m^2)
             "status", 'good',  # good, newDevelopment, renew
             "exterior", True,  # boolean
             "bathrooms", 2,  # integer
             "parkingSpaceHasParkingSpace", True,  # boolean
             "parkingSpaceIsParkingSpaceIncludedInPrice", True,  # boolean
             "neighborhood", 'SantAndreu']  # 'HortaGuinardo', 'Gracia', 'SarriaSantGervasi', 'NouBarris', 'Eixample', 'SantsMontjuic', 'SantMarti', 'LesCorts', 'CiutatVella', 'SantAndreu'

realPrice2 = 350000
instance2 = ["Floor", 5.0,  # float
             "hasLift", True,  # boolean
             "detailedTypeTypology", 'flat',  # countryHouse, duplex, independantHouse, penthouse, ?
             "detailedTypeSubTypology", '?',  # countryHouse, duplex, independantHouse, penthouse, terracedHouse, ?
             "newDevelopment", False,  # boolean
             "rooms", 4,  # integer
             "size", 84,  # integer (m^2)
             "status", 'good',  # good, newDevelopment, renew
             "exterior", True,  # boolean
             "bathrooms", 1,  # integer
             "parkingSpaceHasParkingSpace", True,  # boolean
             "parkingSpaceIsParkingSpaceIncludedInPrice", True,  # boolean
             "neighborhood", 'ViladeGracia']  # 'HortaGuinardo', 'Gracia', 'SarriaSantGervasi', 'NouBarris', 'Eixample', 'SantsMontjuic', 'SantMarti', 'LesCorts', 'CiutatVella', 'SantAndreu'

instances = [instance1, instance2]
realPrices = [realPrice1, realPrice2]

for inst in instances:
    if inst[2] == 'penthouse' or inst[3] == 'penthouse':
        inst[0] = 'subtop'

    if inst[2] == 'independantHouse' or inst[3] == 'terracedHouse' or inst[3] == 'chalet':
        inst[0] = 'top'

# extract the average m^2 price of each neighborhood and district
ngPrices = []; dtPrices = []; districtNames = []; neighbourNames = []
for dt in districtNeighborhood:
    districtNames.append(dt)
    dtPrices.append(smDistrictPrice[dt])

    for neigh in districtNeighborhood[dt]:
        neighbourNames.append(neigh)
        ngPrices.append(smNeighbourhoodPrice[neigh])

dataHeaders = ["Floor", "hasLift", "detailedTypeTypology", "detailedTypeSubTypology", "newDevelopment", "rooms", "size", "status", "exterior", "bathrooms", "parkingSpaceHasParkingSpace",
               "parkingSpaceIsParkingSpaceIncludedInPrice", "districtPrice_m2"]

dataHeadersAmpli = ["Floor", "hasLift", "detailedTypeTypology", "detailedTypeSubTypology", "newDevelopment", "rooms", "size", "status", "exterior", "bathrooms", "parkingSpaceHasParkingSpace",
               "parkingSpaceIsParkingSpaceIncludedInPrice", "districtPrice_m2", "neighborhoodPrice_m2"]

#################### REGRESSORS ########################
# names of the regressors
names = ('MLPReg',
         'Gradient boosting',
         'Extra trees',
         'Random forest',
         'Elastic net',
         'Lasso reg',
         'Ridge reg',
         'Bayesian ridge',
         'SVR linear',
         'KNNRegressor')

# regressors sklearn models
regressors = (MLPRegressor(solver='adam', batch_size=1, hidden_layer_sizes=(15,)),
               GradientBoostingRegressor(n_estimators=100, loss='ls', learning_rate=0.1),
               ExtraTreesRegressor(max_features=None, n_estimators=15),
               RandomForestRegressor(max_features=None, criterion='mse', max_depth=None),
               linear_model.ElasticNet(alpha=0.1),
               linear_model.Lasso(alpha=0.1),
               linear_model.Ridge(alpha=10, solver='sparse_cg'),
               linear_model.BayesianRidge(alpha_2=2e-06, lambda_1=2e-06, lambda_2=5e-07, alpha_1=5e-07),
               SVR(epsilon=0.05, C=10, kernel='rbf'),
               KNeighborsRegressor(n_neighbors=10, weights='distance'))

# Param grids
param_grid_GradientBoosting = {"loss": ['ls', 'lad', 'huber', 'quantile'],
              "learning_rate": [0.1, 0.3],
              "n_estimators": [100, 150]}

param_grid_ExtraTrees = {"n_estimators": [5, 10, 15],
              "max_features": ['auto', None]}  #,
              # "max_features": ['auto', 'sqrt', 'log2', None],
              # "min_samples_split": [2, 3, 10],
              # "min_samples_leaf": [1, 3, 10],
              # "criterion": ["mse", "mae"]}

param_grid_RandomForest = {"max_depth": [3, None],
                          "max_features": ['auto', None],
                          # "max_features": ['auto', 'sqrt', 'log2', None],
                          # "min_samples_split": [2, 3, 10],
                          # "min_samples_leaf": [1, 3, 10],
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

param_grid_MLP = {"maximum_iterations": [200, 300],
                  "hidden_layer_sizes": [(15,), (20,), (100,)],
    #"activation": ['identity', 'logistic', 'tanh', 'relu'],
    "solver": ['lbfgs', 'sgd', 'adam'],
    #"learning_rate": ['constant', 'invscaling', 'adaptive'],
    "batch_size": [1, 'auto']}  # ,
    # "momentum": [0.8, 0.9, 1.0]}

param_grid_KNN = {"n_neighbors": [1, 5, 10],
                  "weights": ['uniform', 'distance']}

param_grids = (param_grid_MLP,
               param_grid_GradientBoosting,
               param_grid_ExtraTrees,
               param_grid_RandomForest,
               param_grid_ElasticNet,
               param_grid_Lasso,
               param_grid_Ridge,
               param_grid_BayesianRidge,
               param_grid_SVR,
               param_grid_KNN)

#########################################################

################## PREDICTIONS ##########################

############# PREDICTION ONLY WITH IT'S DISTRICT DATA ##############
for manualInstIndex in range(len(instances)):
    manualInst = instances[manualInstIndex]
    # continue
    X = []; Y = []
    for dt in districtNames:
        if manualInst[-1] in districtNeighborhood[dt]:
            for instData, instLabel in dataByDistrict[dt]:
                X.append(instData)
                Y.append(instLabel)
            X.append(manualInst[1:len(manualInst):2][:-1])

            rang = 'pos'  # 'posNeg'
            newSelectedDataDataProcessed, scaledY, newNewDataHeaders = parserImproved(X, Y, rang, dataHeaders, scaling=True, normalizing=True, unicolumn=False)

            newTestData = []
            for inst in range(len(newSelectedDataDataProcessed) - 1, len(newSelectedDataDataProcessed) - 2, -1):
                newTestData.append(newSelectedDataDataProcessed[inst])

            newDataProcessed = []  # newLabels = []
            for inst in range(len(newSelectedDataDataProcessed) - 1):
                newDataProcessed.append(newSelectedDataDataProcessed[inst])  # newLabels.append(labels[inst])

            bestRegressor = -1
            bestRegressorAccuracy = 0
            bestParams =[]
            print "\nPREDICTIONS OF INSTANCE {} ONLY WITH IT'S DISTRICT DATA".format(manualInstIndex)
            print "Accuracy with definitive model:"
            for reg, name, regCount, param_grid in zip(regressors, names, range(len(regressors)), param_grids):

                # gridReg = GridSearchCV(reg, param_grid)
                # gridReg.fit(newDataProcessed, Y)
                # bestParams.append(gridReg.best_params_)

                iniTime = time.time()
                cv_results = cross_val_score(reg, newDataProcessed, scaledY, cv=10, scoring='neg_mean_squared_error')
                cv_results_mean = round(np.mean(cv_results), 3) * -100
                finalTime = time.time() - iniTime
                print "Results with: {} Mean squared error: {}% Time: {}".format(name, cv_results_mean, finalTime)

                if 100 - cv_results_mean > bestRegressorAccuracy:
                    bestRegressor = regCount
                    # bestParams = gridReg.best_params_

            print "\nAccuracy with definitive models and data manually introduced:"

            for reg, name, regCount, param_grid in zip(regressors, names, range(len(regressors)), param_grids):
                iniTime = time.time()
                reg.fit(newDataProcessed, Y)
                predictions = reg.predict(newTestData)
                finalTime = time.time() - iniTime
                print "Results with: {} Prediction: {} Time: {}".format(name, predictions * currentPib / initialPib, finalTime)

                if regCount == bestRegressor:
                    bestPredictions = predictions * currentPib / initialPib

            print "\nBEST PREDICTIONS with best model {} and manually introduced data: {} error: {}% with params: {}".format(names[bestRegressor], bestPredictions, abs(bestPredictions - realPrices[manualInstIndex]) / realPrices[manualInstIndex] * 100,  bestParams)

############# PREDICTION ONLY WITH IT'S NEIGBORHOOD DATA ##############
for manualInstIndex in range(len(instances)):
    manualInst = instances[manualInstIndex]
    X = []; Y = []
    for neigh in neighbourNames:
        if manualInst[-1] == neigh:
            for instData, instLabel in dataByNeighborhood[neigh]:
                X.append(instData)
                Y.append(instLabel)
            X.append(manualInst[1:len(manualInst):2][:-1])

            rang = 'pos'  # 'posNeg'
            newSelectedDataDataProcessed, scaledY, newNewDataHeaders = parserImproved(X, Y, rang, dataHeaders, scaling=True, normalizing=True, unicolumn=False)

            newTestData = []
            for inst in range(len(newSelectedDataDataProcessed) - 1, len(newSelectedDataDataProcessed) - 2, -1):
                newTestData.append(newSelectedDataDataProcessed[inst])

            newDataProcessed = []  # newLabels = []
            for inst in range(len(newSelectedDataDataProcessed) - 1):
                newDataProcessed.append(newSelectedDataDataProcessed[inst])  # newLabels.append(labels[inst])

            bestRegressor = -1
            bestRegressorAccuracy = 0
            print "\nPREDICTIONS OF INSTANCE {} ONLY WITH IT'S NEIGHBORHOOD DATA".format(manualInstIndex)
            print "Accuracy with definitive model:"
            for reg, name, regCount, param_grid in zip(regressors, names, range(len(regressors)), param_grids):
                # gridReg = GridSearchCV(reg, param_grid)
                # gridReg.fit(newDataProcessed, Y)

                iniTime = time.time()
                cv_results = cross_val_score(reg, newDataProcessed, scaledY, cv=10, scoring='neg_mean_squared_error')
                cv_results_mean = round(np.mean(cv_results), 3) * -100
                finalTime = time.time() - iniTime
                print "Results with: {} Mean squared error: {}% Time: {}".format(name, cv_results_mean, finalTime)

                if 100 - cv_results_mean > bestRegressorAccuracy:
                    bestRegressor = regCount
                    # bestParams = gridReg.best_params_

            print "\nAccuracy with definitive models and data manually introduced:"

            for reg, name, regCount, param_grid in zip(regressors, names, range(len(regressors)), param_grids):
                iniTime = time.time()
                reg.fit(newDataProcessed, Y)
                predictions = reg.predict(newTestData)
                finalTime = time.time() - iniTime
                print "Results with: {} Prediction: {} Time: {}".format(name, predictions * currentPib / initialPib, finalTime)

                if regCount == bestRegressor:
                    bestPredictions = predictions * currentPib / initialPib

            print "\nBEST PREDICTIONS with best model {} and manually introduced data: {} error: {} with params: {}".format(names[bestRegressor], bestPredictions, abs(bestPredictions - realPrices[manualInstIndex]) / realPrices[manualInstIndex] * 100, bestParams)

############# PREDICTION WITH ALL BARCELONA DATA ##############
for manualInstIndex in range(len(instances)):
    manualInst = instances[manualInstIndex]
    X = []; Y = []
    for neigh in neighbourNames:
        if manualInst[-1] == neigh:
            dtPrice = smDistrictPrice[dt]
            neighPrice = smNeighbourhoodPrice[neigh]
            for instData, instLabel in dataByNeighborhood[neigh]:
                newInstData = []
                newManualInst = []
                for field in instData:
                    newInstData.append(field)
                newInstData.append(dtPrice)
                newInstData.append(neighPrice)

                X.append(newInstData)
                Y.append(instLabel)

            for field in manualInst[1:len(manualInst):2][:-1]:
                newManualInst.append(field)
            newManualInst.append(dtPrice)
            newManualInst.append(neighPrice)

            X.append(newManualInst)

            rang = 'pos'  # 'posNeg'
            newSelectedDataDataProcessed, scaledY, newNewDataHeaders = parserImproved(X, Y, rang, dataHeadersAmpli, scaling=True, normalizing=True, unicolumn=False)

            newTestData = []
            for inst in range(len(newSelectedDataDataProcessed) - 1, len(newSelectedDataDataProcessed) - 2, -1):
                newTestData.append(newSelectedDataDataProcessed[inst])

            newDataProcessed = []  # newLabels = []
            for inst in range(len(newSelectedDataDataProcessed) - 1):
                newDataProcessed.append(newSelectedDataDataProcessed[inst])  # newLabels.append(labels[inst])

            bestRegressor = -1
            bestRegressorAccuracy = 0
            print "\nPREDICTIONS OF INSTANCE {} PREDICTION WITH ALL BARCELONA DATA".format(manualInstIndex)
            print "Accuracy with definitive model:"
            for reg, name, regCount, param_grid in zip(regressors, names, range(len(regressors)), param_grids):
                # gridReg = GridSearchCV(reg, param_grid)
                # gridReg.fit(newDataProcessed, Y)

                iniTime = time.time()
                cv_results = cross_val_score(reg, newDataProcessed, scaledY, cv=10, scoring='neg_mean_squared_error')
                cv_results_mean = round(np.mean(cv_results), 3) * -100
                finalTime = time.time() - iniTime
                print "Results with: {} Mean squared error: {}% Time: {}".format(name, cv_results_mean, finalTime)

                if 100 - cv_results_mean > bestRegressorAccuracy:
                    bestRegressor = regCount
                    # bestParams = gridReg.best_params_

            print "\nAccuracy with definitive models and data manually introduced:"

            for reg, name, regCount, param_grid in zip(regressors, names, range(len(regressors)), param_grids):
                iniTime = time.time()
                reg.fit(newDataProcessed, Y)
                predictions = reg.predict(newTestData)
                finalTime = time.time() - iniTime
                print "Results with: {} Prediction: {} Time: {}".format(name, predictions * currentPib / initialPib, finalTime)

                if regCount == bestRegressor:
                    bestPredictions = predictions * currentPib / initialPib

            print "\nBEST PREDICTIONS with best model {} and manually introduced data: {} error: {} with params: {}".format(names[bestRegressor], bestPredictions, abs(bestPredictions - realPrices[manualInstIndex]) / realPrices[manualInstIndex] * 100, bestParams)

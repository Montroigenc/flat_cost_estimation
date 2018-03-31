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
from sklearn.model_selection import KFold # import KFold
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import pickle
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel

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
names = ('Gradient boosting',
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
data = []
labels = []
rang = 'pos' #'posNeg'
for dt in districtNames:
    X = []; Y = []
    dtPrice = smDistrictPrice[dt]
    for inst in dataByDistrict[dt]:
        inst[0].append(dtPrice)
        X.append(inst[0])
        Y.append(inst[1])

    X, Y = minMaxScaling(X, Y, rang, dataHeaders)  # scale all features but "districtPrice_m2"

    for instIndex in range(len(X)):
        data.append(X[instIndex])
        labels.append(Y[instIndex])

preProcessedX, preProcessedY, newDataHeaders = parserImproved(data, labels, rang, dataHeaders, scaling=True, normalizing=True, unicolumn=False)
# instancestotxt(preProcessedX, preProcessedY)

################### Build a forest and compute the FEATURE IMPORTANCE ############################333
forest = ExtraTreesRegressor(n_estimators=250, random_state=0)

forest.fit(preProcessedX, preProcessedY)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(len(preProcessedX[1])):
    print("{}. feature {} ({})".format(f + 1, newDataHeaders[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(preProcessedX[1])), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(preProcessedX[1])), indices)
plt.xlim([-1, len(preProcessedX[1])])
plt.show()
#
# ################################################################################

# split in training and testing data
X_train, X_test, y_train, y_test = train_test_split(preProcessedX, preProcessedY, test_size=0.3)

print "\nRESUTLTS"

for reg, name in zip(classifiers, names):
    cv_results = cross_val_score(reg, preProcessedX, preProcessedY, cv=10)
    cv_results_mean = round(np.mean(cv_results), 3)

    startTime = time.clock()
    reg.fit(X_train, y_train)
    trainAccuracy = round(reg.score(X_train, y_train), 3)
    testAccuracy = round(reg.score(X_test, y_test), 3)
    endTime = round((time.clock() - startTime) * 1000, 3)
    predictions = reg.predict(X_test)
    testError = np.abs(y_test - predictions)

    accuracy = np.ones((len(testError))) - testError
    meanAccuracy = round(np.mean(accuracy), 3)
    maxAccuracy = round(np.max(accuracy), 3)
    minAccuracy = round(np.min(accuracy), 3)

    print "regressor: {:^20} cv: {:<5} trainAcc {:<5} testAcc {:<5} Acc: (mean: {:<5} max: {:<5} min: {:<5}  computing time: {}".format(name, cv_results_mean, trainAccuracy, testAccuracy, meanAccuracy, maxAccuracy, minAccuracy, endTime)


# feature selection
print "\nRESUTLTS WITH FEATURE SELECTION"
model = SelectFromModel(forest, prefit=True)
preProcessedFeatureSelectedX = model.transform(preProcessedX)

selectedIndices = model.get_support(indices=True)

selectedFeatures = ""
for ind in selectedIndices:
    selectedFeatures += newDataHeaders[ind] + " "

print "Selected features: {}".format(selectedFeatures)

# split in training and testing data
X_train, X_test, y_train, y_test = train_test_split(preProcessedFeatureSelectedX, preProcessedY, test_size=0.3)


for reg, name in zip(classifiers, names):
    cv_results = cross_val_score(reg, preProcessedFeatureSelectedX, preProcessedY, cv=10)
    cv_results_mean = round(np.mean(cv_results), 3)

    startTime = time.clock()
    reg.fit(X_train, y_train)
    trainAccuracy = round(reg.score(X_train, y_train), 3)
    testAccuracy = round(reg.score(X_test, y_test), 3)
    endTime = round((time.clock() - startTime) * 1000, 3)
    predictions = reg.predict(X_test)
    testError = np.abs(y_test - predictions)

    accuracy = np.ones((len(testError))) - testError
    meanAccuracy = round(np.mean(accuracy), 3)
    maxAccuracy = round(np.max(accuracy), 3)
    minAccuracy = round(np.min(accuracy), 3)

    print "regressor: {:^20} cv: {:<5} trainAcc {:<5} testAcc {:<5} Acc: (mean: {:<5} max: {:<5} min: {:<5}  computing time: {}".format(name, cv_results_mean, trainAccuracy, testAccuracy, meanAccuracy, maxAccuracy, minAccuracy, endTime)





pass

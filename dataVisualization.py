from __future__ import division
from data_preprocessing import *
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

output = open('smNeighbourhoodnumber.pkl', 'rb'); smNeighbourhoodnumber = pickle.load(output); output.close()
output = open('smNeighbourhoodPrice.pkl', 'rb'); smNeighbourhoodPrice = pickle.load(output); output.close()
output = open('neighbourhoodDistrict.pkl', 'rb'); neighbourhoodDistrict = pickle.load(output); output.close()
output = open('smDistrictPrice.pkl', 'rb'); smDistrictPrice = pickle.load(output); output.close()
output = open('smDistrictnumber.pkl', 'rb'); smDistrictnumber = pickle.load(output); output.close()
output = open('dataByNeighborhood.pkl', 'rb'); dataByNeighborhood = pickle.load(output); output.close()
output = open('dataByDistrict.pkl', 'rb'); dataByDistrict = pickle.load(output); output.close()

dataHeaders = ["Floor", "hasLift", "detailedTypeTypology", "detailedTypeSubTypology", "newDevelopment", "rooms", "size", "status", "exterior", "bathrooms", "parkingSpaceHasParkingSpace", "parkingSpaceIsParkingSpaceIncludedInPrice", "districtPrice_m2"]

########## PLOT INFO OF EACH NEIGHBORHOOD BASED ON m^2 PRICE ##########
# classify each neighbourhood by district
districtNeighborhood = defaultdict(list)
for neigh in neighbourhoodDistrict:
    districtNeighborhood[neighbourhoodDistrict[neigh]].append(neigh)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
colorIndex = 0
xPrice = []; ngPrices = []; dtPrices = []; districtNames = []; neighbourNames = []; districtColors = []
neighColors = ""
for dist in districtNeighborhood:
    color = colors[colorIndex]
    districtNames.append(dist)
    districtColors.append(color)
    colorIndex += 1
    if colorIndex >= len(colors) - 1:
        colorIndex = 0

    for neigh in districtNeighborhood[dist]:
        neighbourNames.append(neigh)
        xPrice.append(neigh + '(' + dist + ')')
        ngPrices.append(smNeighbourhoodPrice[neigh])
        dtPrices.append(smDistrictPrice[dist])
        neighColors += color

ngPrices = np.asarray(ngPrices)
dtPrices = np.asarray(dtPrices)

s = pd.DataFrame({
 'ngPrices':ngPrices,
 'dtPrices':dtPrices,
}, index=xPrice)

#Plot the data:
ax = s[['ngPrices', 'dtPrices']].plot(kind='bar', color=[neighColors + neighColors],)

# s.plot(kind='bar', color=neighColors,)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.5)
plt.tick_params(axis='x', labelsize=6)
plt.xlabel('neighborhood')
plt.ylabel('price/m^2')
plt.title('price/m^2 by neighborhood')

# plt.legend(districtColors, [districtNames])
ax.legend().set_visible(False)

for i in range(len(neighbourNames)):
 ax.text(i, ax.get_ylim()[1]*.9, smNeighbourhoodnumber[neighbourNames[i]], horizontalalignment='center', fontsize=6, rotation='vertical')

plt.show()
##################################################

########## VISUALIZE ATTRIBUTE RELATIONS ##########
districtNames = dataByDistrict.keys()
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

    for instIndex in range(len(X)):
        data.append(X[instIndex])
        labels.append(Y[instIndex])


npData = np.array(data)
npY = np.array(labels).astype(float)

##########  PLOT INFLUENCES  ########
for featureIndex in range(len(npData[0])):
    featureName = dataHeaders[featureIndex]
    npDataColumn = npData[:, featureIndex]
    rang = "pos";
    scaling = False;
    normalizing = True
    # here we use the parserImproved only to normalize, not scaling the data. Note that unicolumn has to be set to True due that the input dataset (npDataColumn)
    # has only one column
    x, y, newDataHeaders = parserImproved(npDataColumn, npY, rang, dataHeaders[featureIndex], scaling=False, normalizing=True, unicolumn=True)

    npX = []
    if len(x[0]) > 1:
        x = x * np.array([range(len(x[0])), ]*len(x))
        for i in range(len(x)):
            npX.append(sum(x[i]))

        newDataHeaders = np.array(newDataHeaders)[:, 1]

        plt.xticks(range(len(x[0])), newDataHeaders, rotation=90)
        plt.subplots_adjust(bottom=0.3)
        plt.tick_params(axis='x', labelsize=8)

    else:
        for i in range(len(x)):
            npX.append(x[i][0])

    x = np.array(npX).astype(float)
    p = np.poly1d(np.polyfit(x, y, 1))
    plt.plot(x, y, '.', x, p(x), '--k')

    plt.xlabel(featureName)
    plt.ylabel('price')
    plt.title("price - {}".format(featureName))
    plt.show()
    pass
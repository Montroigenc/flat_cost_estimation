from __future__ import division
import numpy as np
# import math

def parserImproved(X, Y, rang, dataHeaders, scaling, normalizing, unicolumn):

    newDatamatHeaders = []
    # Transforms the list into numpy arrays (in order to work better with them)
    datamat = np.array(X)
    classes = np.array(Y)

    if normalizing:
        # This erases useless features and adds values to no-data elements
        if unicolumn:
            headers = []
            headers.append(dataHeaders)
            # Find the most common value in the column 'col'
            currentCol = datamat
            uniques, counts = np.unique(currentCol, return_counts=True)
            mostcommon = uniques[np.argmax(counts)]

            # Executes the first or the second depending on the datatype of col
            try:  # If col is an array of numbers, substitutes the 'nan' (no data) for the average
                if 'top' in uniques:  # top mean the highest floor of a building, so we assign to them the highest floor in the dataset
                    sortedUniques = np.flip(sorted(uniques), 0)
                    highestFloor = -1
                    for maxIndex in range(len(sortedUniques)):
                        try:
                            if float(sortedUniques[maxIndex]) > highestFloor:
                                highestFloor = float(sortedUniques[maxIndex])
                        except:
                            pass

                    currentCol[currentCol == 'subtop'] = highestFloor + 1
                    currentCol[currentCol == 'top'] = highestFloor + 2
                    currentCol[currentCol == '?'] = highestFloor / 2
                    uniques = uniques[uniques != 'top']
                    uniques = uniques[uniques != 'subtop']
                    uniques = uniques[uniques != '?']

                else:
                    if '?' in uniques:
                        index = currentCol == '?'
                        if mostcommon == '?':  # in the strange case mostcommon is "?", we pick-up the second mostcommon value
                            mostcommon = uniques[np.flip(np.argsort(counts), 0)[1]]
                        currentCol[index] = mostcommon
                        uniques = uniques[uniques != '?']

                index_nan = currentCol == 'nan'
                index_nonan = currentCol != 'nan'

                average = np.average(currentCol[index_nonan].astype(np.float))
                currentCol[index_nan] = average
                currentCol = np.asarray(currentCol).astype(float)

                auxCol = np.zeros((len(currentCol), 1))
                for i in range(len(currentCol)):
                    auxCol[i, 0] = currentCol[i]
                currentCol = auxCol

            except:
                if len(uniques) == 1:  # If there is only one value in this feature, delete column
                    headers = []
                    currentCol = []
                elif len(uniques) == 2:
                    index = currentCol == uniques[0]
                    currentCol[index] = 0.0
                    index = currentCol == uniques[1]
                    currentCol[index] = 1.0

                    auxCol = np.zeros((len(currentCol), 1))
                    for i in range(len(currentCol)):
                        auxCol[i, 0] = currentCol[i]
                    currentCol = auxCol
                else:
                    originalHeader = headers[0]
                    headers = []
                    expandCol = np.zeros((len(currentCol), 1))
                    originalCol = currentCol
                    for uni in range(len(uniques)):
                        headers.append([originalHeader, uniques[uni]])
                        if uni > 0:
                            auxExpandCol = np.zeros((len(expandCol), len(expandCol[0]) + 1))
                            auxExpandCol[:, :-1] = expandCol
                            expandCol = auxExpandCol
                        expandCol[:, -1] = originalCol == uniques[uni]
                        currentCol = expandCol
                currentCol = np.asarray(currentCol).astype(float)

            for header in headers:
                newDatamatHeaders.append(header)

            newDatamat = np.zeros_like((currentCol))

            for i in range(len(currentCol)):
                if len(currentCol[0]) > 1:
                    for j in range(len(currentCol[0])):
                        newDatamat[i, j] = currentCol[i][j]
                else:
                    newDatamat = currentCol
                    break
        else:
            for col in range(len(datamat[0])):
                headers = []
                headers.append(dataHeaders[col])
                # Find the most common value in the column 'col'
                currentCol = datamat[:, col]
                uniques, counts = np.unique(currentCol, return_counts=True)
                mostcommon = uniques[np.argmax(counts)]

                # Executes the first or the second depending on the datatype of col
                try:  # If col is an array of numbers, substitutes the 'nan' (no data) for the average
                    if 'top' in uniques:  # top mean the highest floor of a building, so we assign to them the highest floor in the dataset
                        sortedUniques = np.flip(sorted(uniques), 0)
                        highestFloor = -1
                        for maxIndex in range(len(sortedUniques)):
                            try:
                                if float(sortedUniques[maxIndex]) > highestFloor:
                                    highestFloor = float(sortedUniques[maxIndex])
                            except:
                                pass

                        currentCol[currentCol == 'subtop'] = highestFloor + 1   #  in chase of penthouses we assign a floor over the highest, due that usually the higher the flat, the higher the price,
                                                                                # and it has no sense to penalize a penthouse of a building of 3 floors above a normal flat in the middle of a 10 floors building
                        currentCol[currentCol == 'top'] = highestFloor + 2      #  in chase of chalets we assign a floor 2 floors over the highest, due that if we assign the 0 floor, they are mixed with cheap flats
                        currentCol[currentCol == '?'] = highestFloor / 2
                        uniques = uniques[uniques != 'top']
                        uniques = uniques[uniques != 'subtop']
                        uniques = uniques[uniques != '?']

                    else:
                        if '?' in uniques:
                            index = currentCol == '?'
                            if mostcommon == '?':  # in the strange case mostcommon is "?", we pick-up the second mostcommon value
                                mostcommon = uniques[np.flip(np.argsort(counts), 0)[1]]
                            currentCol[index] = mostcommon
                            uniques = uniques[uniques != '?']

                    index_nan = currentCol == 'nan'
                    index_nonan = currentCol != 'nan'

                    average = np.average(currentCol[index_nonan].astype(np.float))
                    currentCol[index_nan] = average
                    currentCol = np.asarray(currentCol).astype(float)

                    auxCol = np.zeros((len(currentCol), 1))
                    for i in range(len(currentCol)):
                        auxCol[i, 0] = currentCol[i]
                    currentCol = auxCol

                except:
                    if len(uniques) == 1:  # If there is only one value in this feature, delete column
                        headers = []
                        currentCol = []
                    elif len(uniques) == 2:
                        index = currentCol == uniques[0]
                        currentCol[index] = 0.0
                        index = currentCol == uniques[1]
                        currentCol[index] = 1.0

                        auxCol = np.zeros((len(currentCol), 1))
                        for i in range(len(currentCol)):
                            auxCol[i, 0] = currentCol[i]
                        currentCol = auxCol
                    else:
                        originalHeader = headers[0]
                        headers = []
                        expandCol = np.zeros((len(currentCol), 1))
                        originalCol = currentCol
                        for uni in range(len(uniques)):
                            headers.append([originalHeader, uniques[uni]])
                            if uni > 0:
                                auxExpandCol = np.zeros((len(expandCol), len(expandCol[0]) + 1))
                                auxExpandCol[:, :-1] = expandCol
                                expandCol = auxExpandCol
                            expandCol[:, -1] = originalCol == uniques[uni]
                            currentCol = expandCol
                    currentCol = np.asarray(currentCol).astype(float)

                for header in headers:
                    newDatamatHeaders.append(header)

                if col > 0:
                    if len(currentCol) > 0:
                        newDatamat = np.concatenate((newDatamat, currentCol), axis=1)
                else:
                    newDatamat = np.zeros((len(currentCol), len(currentCol[0])))

                    for i in range(len(currentCol)):
                        for j in range(len(currentCol[0])):
                            newDatamat[i, j] = currentCol[i][j]
    else:
        newDatamat = datamat
        newDatamatHeaders = dataHeaders

    if scaling:  # in case scaling is True, we scale the data and labels to the selected range [0,1] or [-1,1]

        datamat_list = newDatamat.tolist()
        for col in range(0, newDatamat.shape[1]):
            col_float = newDatamat[:, col].astype(np.float)
            if np.max(col_float)-np.min(col_float) == 0:
                col_float_normalized = np.zeros(len(newDatamat[:, col])).astype(float)
            else:
                if rang == 'posNeg':  # With this normalization we get the values so that the maximum is always 1 and the minumum -1
                    col_float_mean_extracted = col_float - np.mean(col_float)
                    col_float_normalized = np.divide(col_float_mean_extracted, np.max([np.max(col_float_mean_extracted),np.abs(np.min(col_float_mean_extracted))]))
                elif rang == 'pos':  # With this normalization we get the values so that the maximum is always 1 and the minumum 0
                    col_float_normalized = (col_float - np.min(col_float)) / (np.max(col_float) - np.min(col_float))
                else:
                    raise

            for row in range(0, newDatamat.shape[0]):
                datamat_list[row][col] = col_float_normalized[row]

        # Find the most common value in the column 'col'
        uniques, counts = np.unique(classes[:], return_counts=True)
        # Executes the first or the second depending on the datatype of col
        try:  # Check if there are categorical labels
            classes = float(classes)
        except:  # There are categorical labels
            for uni in range(len(uniques)):
                index = classes == uniques[uni]
                classes[index] = uni

        col_float = classes[:].astype(np.float)
        if rang == 'posNeg':  # With this normalization we get the values so that the maximum is always 1 and the minumum -1
            classes_mean_extracted = col_float - np.mean(col_float)
            classes_normalized = np.divide(classes_mean_extracted, np.max([np.max(classes_mean_extracted), np.abs(np.min(classes_mean_extracted))]))
        elif rang == 'pos':  # With this normalization we get the values so that the maximum is always 1 and the minumum 0
            classes_normalized = (col_float - np.min(col_float)) / (np.max(col_float) - np.min(col_float))
        else:
            raise

        classes_list = classes_normalized.tolist()
    else:
        datamat_list = newDatamat
        classes_list = classes

    return datamat_list, classes_list, newDatamatHeaders

def minMaxScaling(X, Y, rang, dataHeaders):  # this function is exclusive to scale labels and all data features but the districtPrice_m2 feature
                                             # this is because we would like to preserve the unequality of scaling of the districtPrice_m2 feature
                                             # between the different districts. Once all districts data is in a unique dataset, then we will normalize
                                             # again all dataset, districtPrice_m2 included.

    # Transforms the list into numpy arrays (in order to work better with them)
    newDatamat = np.array(X)
    classes = np.array(Y)

    "districtPrice_m2"

    # Numerical data should be normalized
    datamat_list = newDatamat.tolist()
    for col in range(0, newDatamat.shape[1]):
        if dataHeaders[col] == "districtPrice_m2":
            continue
        try:
            col_float = newDatamat[:, col].astype(np.float)
            if np.max(col_float)-np.min(col_float) == 0:
                col_float_normalized = np.zeros(len(newDatamat[:, col])).astype(float)
            else:
                if rang == 'posNeg':  # With this normalization we get the values so that the maximum is always 1 and the minumum -1
                    col_float_mean_extracted = col_float - np.mean(col_float)
                    col_float_normalized = np.divide(col_float_mean_extracted, np.max([np.max(col_float_mean_extracted),np.abs(np.min(col_float_mean_extracted))]))
                elif rang == 'pos':  # With this normalization we get the values so that the maximum is always 1 and the minumum 0
                    col_float_normalized = (col_float - np.min(col_float)) / (np.max(col_float) - np.min(col_float))
                else:
                    raise

            for row in range(0, newDatamat.shape[0]):
                datamat_list[row][col] = col_float_normalized[row]

        except:
            pass

    # Find the most common value in the column 'col'
    uniques, counts = np.unique(classes[:], return_counts=True)
    # Executes the first or the second depending on the datatype of col
    try:  # Check if there are categorical labels
        classes = float(classes)
    except:  # There are categorical labels
        for uni in range(len(uniques)):
            index = classes == uniques[uni]
            classes[index] = uni

    col_float = classes[:].astype(np.float)
    if rang == 'posNeg':  # With this normalization we get the values so that the maximum is always 1 and the minumum -1
        classes_mean_extracted = col_float - np.mean(col_float)
        classes_normalized = np.divide(classes_mean_extracted, np.max([np.max(classes_mean_extracted), np.abs(np.min(classes_mean_extracted))]))
    elif rang == 'pos':  # With this normalization we get the values so that the maximum is always 1 and the minumum 0
        classes_normalized = (col_float - np.min(col_float)) / (np.max(col_float) - np.min(col_float))
    else:
        raise

    classes_list = classes_normalized.tolist()

    return datamat_list, classes_list





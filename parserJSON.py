# -*- coding: utf-8 -*-
from __future__ import division
import unicodedata
import string
import pickle
import os
from collections import defaultdict

"""
This script is in charge of traduce JSON homes in instances (matrix)
Properties:
 - Floor
 - PropertyType
 - m2
 - exterior
 - bedrooms
 - bathrooms
 - district
 - status
 - newDevelopment
 - hasLift
 - parking - is included in the price?
 - areaprice (we have to add it manually)
 - finalPrice
"""

import json
import glob

def JSONparser(file, smNeighbourhoodnumber, smNeighbourhoodPrice, smDistrictnumber, smDistrictPrice, neighbourhoodDistrict, dataByNeighborhood, dataByDistrict):
    """
    From JSON to memory
    :param file: relative path
    :return: instances and labels
    """
    data = json.load(open(file))
    elements = data["elementList"]
    instanceList = []
    priceList = []

    for i in elements:
        homeproperties = []
        #trueValues = i.values()
        # Si el inmueble pertenece a la ciudad de Barcelona
        if (i["municipality"] == 'Barcelona'):
            try:
                if 'country' in i["detailedType"]["typology"] or 'country' in i["detailedType"]["subTypology"]:
                    pass
            except:
                pass

            # Labels
            priceList.append(i['price'])

            # Attributes
            try:
                # penthouse mean that it is in the top of the building, we will process that kind of accommodation in later steps
                if asciiAdapt(i["detailedType"]["typology"]) == 'penthouse' or asciiAdapt(i["detailedType"]["subTypology"]) == 'penthouse':
                    floor = 'subtop'
                elif i["propertyType"] == 'chalet' or asciiAdapt(i["detailedType"]["subTypology"]) == 'independantHouse' or asciiAdapt(i["detailedType"]["subTypology"]) == 'terracedHouse':
                    floor = 'top'
                else:
                    floor = float(i["floor"])
                    # if int(i["floor"]) >= 1.0:
                    #     floor = 1.0
                    # else:
                    #     floor = float(i["floor"])
            except:
                try:
                    if i["floor"] == 'bj':  #-> Bajo exterior
                        floor = 0.0
                    elif i["floor"] == 'en': #-> entresuelo
                        floor = 0.5
                    elif i["floor"] == 'ss': #-> semi-sotano
                        floor = -0.5
                    elif i["floor"] == 'st': #-> sotano
                        floor = -1.0
                    else:
                        floor = float(i["floor"])
                except:
                    floor = "?"  # it's a flat but floor is not defined

            homeproperties.append(floor)

            try:
                lift = ""
                if i["propertyType"] == 'chalet':  # we will treat chalets as if they have lift due that they don't need it
                    lift = True
                else:
                    lift = i["hasLift"]
                if not i["hasLift"]:
                    pass
            except:
                if lift == "":
                    try:
                        lift = i["hasLift"]
                    except:
                        lift = False
            homeproperties.append(lift)
            # homeproperties.append(asciiAdapt(i["district"]))
            district = asciiAdapt(i["district"])

            if district in smDistrictPrice:
                smDistrictnumber[district] += 1
                smDistrictPrice[district] = (smDistrictPrice[district] + i["priceByArea"])
            else:
                smDistrictnumber[district] = 1
                smDistrictPrice[district] = i["priceByArea"]

            neigborhood = asciiAdapt(i["neighborhood"])
            homeproperties.append(neigborhood)

            if neigborhood in smNeighbourhoodPrice:
                smNeighbourhoodnumber[neigborhood] += 1
                smNeighbourhoodPrice[neigborhood] = (smNeighbourhoodPrice[neigborhood] + i["priceByArea"])
            else:
                smNeighbourhoodnumber[neigborhood] = 1
                smNeighbourhoodPrice[neigborhood] = i["priceByArea"]
                neighbourhoodDistrict[neigborhood] = district

            try:
                homeproperties.append(asciiAdapt(i["detailedType"]["typology"]))
            except:
                homeproperties.append("?")
                # homeproperties.append("flat")

            try:
                homeproperties.append(asciiAdapt(i["detailedType"]["subTypology"]))
            except:
                # homeproperties.append("flat")
                homeproperties.append("?")


            # homeproperties.append(i["propertyType"])
            homeproperties.append(i["newDevelopment"])
            # homeproperties.append(i["priceByArea"]) #Este atributo es trampa, es el precio del m2 del inmueble

            try:
                if asciiAdapt(i["detailedType"]["typology"]) == 'studio' or asciiAdapt(i["detailedType"]["subTypology"]) == 'studio':
                    rooms = '?'  # here we put the studios to be considered as to have the half of the maximum room number due that it doesn't have sense
                                 # to take them into account like if they have no rooms on only 1 room
                else:
                    rooms = i["rooms"]
            except:
                rooms = i["rooms"]  # rooms are always defined

            homeproperties.append(rooms)

            homeproperties.append(i["size"])
            try:
                homeproperties.append(asciiAdapt(i["status"]))
            except:
                homeproperties.append("?")

            homeproperties.append(i["exterior"])
            homeproperties.append(i["bathrooms"])

            try:
                homeproperties.append(i["parkingSpace"]["hasParkingSpace"])
            except:
                homeproperties.append("False")

            try:
                homeproperties.append(i["parkingSpace"]["isParkingSpaceIncludedInPrice"])
            except:
                homeproperties.append("False")

            instanceList.append(homeproperties)
            homeproperties.remove(neigborhood)
            dataByNeighborhood[neigborhood].append([homeproperties, float(i["price"])])
            dataByDistrict[district].append([homeproperties, float(i["price"])])


    return instanceList, priceList, smNeighbourhoodnumber, smNeighbourhoodPrice, smDistrictnumber, smDistrictPrice, neighbourhoodDistrict, dataByNeighborhood, dataByDistrict

def asciiAdapt(ud):
    """
    Treatment of input attribute
    :param nd: unicode data
    :return: string(data)
    """
    result = unicodedata.normalize('NFKD', ud).encode('ascii', 'ignore')
    result = result.replace(" ", "")
    result = result.translate(None, string.punctuation)

    return result

def instancestotxt(instances, labels):
    """
    In charge of write instances in a txt file, in order to make the problem portable (Weka)
    :param instances: the return 1 of JSONParser
    :param labels: the return 2 of JSONParser
    :return:
    """
    contador = 0
    with open("./data/aleixFile.txt", "a") as f:
        for instance in instances:
            for atribbute in instance:
                f.write(str(atribbute) + ";")
            f.write(str(labels[contador])+"\n")
            contador += 1


def parserfromtxt(path):
    """
    From txt to memory
    :param path: relative path
    :return: instances and labels
    """
    X = []
    Y = []
    listAux = []
    infile = open(path, 'r')
    for line in infile:
        aux = line.split(";")
        for item in aux[:(len(aux) - 1)]:
            try:
                listAux.append(float(item))
            except:
                listAux.append(item)
        X.append(listAux)
        listAux = []
        Y.append(float(aux[len(aux) - 1]))
    infile.close()
    return X, Y

def allJSONtoTxt():
    # firstly, in case we already have created a previous txt file with data, we delete it
    try:
        os.remove("./data/definitiveInstanceFile.txt")
    except:
        pass

    smNeighbourhoodPrice = {}
    smNeighbourhoodnumber = {}
    smDistrictPrice = {}
    smDistrictnumber = {}
    neighbourhoodDistrict = {}
    dataByDistrict = defaultdict(list)
    dataByNeighborhood = defaultdict(list)

    fich = glob.glob(".\originaldata\*")
    for fi in fich:
        if fi != '.\\originaldata\\desktop.ini':
            instances, labels, smNeighbourhoodnumber, smNeighbourhoodPrice, smDistrictnumber, smDistrictPrice, neighbourhoodDistrict, dataByNeighborhood, dataByDistrict = JSONparser(fi, smNeighbourhoodnumber, smNeighbourhoodPrice, smDistrictnumber, smDistrictPrice, neighbourhoodDistrict, dataByNeighborhood, dataByDistrict)
            instancestotxt(instances, labels)

    for nb in smNeighbourhoodPrice.keys():
        smNeighbourhoodPrice[nb] = smNeighbourhoodPrice[nb] / smNeighbourhoodnumber[nb]

    for dt in smDistrictPrice.keys():
        smDistrictPrice[dt] = smDistrictPrice[dt] / smDistrictnumber[dt]

    output = open('smNeighbourhoodnumber.pkl', 'wb'); pickle.dump(smNeighbourhoodnumber, output); output.close()
    output = open('smNeighbourhoodPrice.pkl', 'wb'); pickle.dump(smNeighbourhoodPrice, output); output.close()
    output = open('neighbourhoodDistrict.pkl', 'wb'); pickle.dump(neighbourhoodDistrict, output); output.close()
    output = open('dataByNeighborhood.pkl', 'wb'); pickle.dump(dataByNeighborhood, output); output.close()
    output = open('dataByDistrict.pkl', 'wb'); pickle.dump(dataByDistrict, output); output.close()
    output = open('smDistrictnumber.pkl', 'wb'); pickle.dump(smDistrictnumber, output); output.close()
    output = open('smDistrictPrice.pkl', 'wb'); pickle.dump(smDistrictPrice, output); output.close()


"""
instances, labels = JSONparser("pruebaJSON.json")
instancestotxt(instances, labels)
X, Y = parserfromtxt("definitiveInstanceFile.txt")
"""

allJSONtoTxt()
#print "finish"

"""
instances, labels, a, b = JSONparser("./originaldata/pagina1.json")
print a
print b
"""
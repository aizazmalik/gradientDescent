import sys
import pandas as pd
from pandas import DataFrame
import math

#Gradient Descent Code

#Reading the dataset from an excel file
#Change the path below to train on your dataset
ReadExcel = pd.read_excel(r,'datasetest.xlsx')

#Reading columns of the dataset (can be changed if required)
df = DataFrame(ReadExcel, columns=['Car', 'Manufacturing Year', 'MeterReading', 'Transmission', 'Registered City',
                                   'Color', 'Assembly', 'Engine Capacity', 'Body Type', 'Price'])

#Storing dataset in a list
dataSet = df.get_values()

#getting m = number of training examples
dataLength = len(dataSet)

#Setting alpha or Step size
alpha = 0.1

#Setting maximum number of iterations for the code
maxIterations = 10000

#Setting the threhold for convergance
threshold = sys.float_info.epsilon

#Function for scaling the features from the dataset provided
def featureScaling(dataSetTemp, dataLength):
    count = 0
    w, h = 10,dataLength;
    tempData = [[0 for x in range(w)] for y in range(h)]

    #Scaling the features x1 to x9 provided in parameters by the dataset
    #Method used for scaling: x = x-min(x) / max(x)-min(x)
    for x in dataSetTemp:
        tempData[count][0] = round((x[0]-1)/(16-1), 4)
        tempData[count][1] = round((x[1]-1990)/(2018-1990), 4)
        tempData[count][2] = round((x[2]-50)/(999000-50), 4)
        tempData[count][3] = round((x[3]-0)/(1-0), 4)
        tempData[count][4] = round((x[4]-1)/(97-1), 4)
        tempData[count][5] = round((x[5]-1)/(15-1), 4)
        tempData[count][6] = round(((x[6]-0)/(1-0)), 4)
        tempData[count][7] = round(((x[7]-600)/(4000-600)), 4)
        tempData[count][8] = round(((x[8]-1)/(5-1)), 4)
        # tempData[count][9] = round((x[9]-0.1)/(18.5-0.1), 4) // Labels will not be scaled
        tempData[count][9] = round(x[9], 4)
        print(count)
        print(tempData[count])
        count = count + 1
    #returning the scaled data set
    return tempData

#Function for calculating the hypothesis based upon values of theta and parameters provided
def calculatehypothesis(theta0, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9,
                        x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return ((theta0*0) + (theta1*x1) + (theta2*x2) + (theta3*x3) + (theta4*x4) + (theta5*x5) + (theta6*x6) + (theta7*x7) +
            (theta8*x8) + (theta9*x9))


#Checking if the model has converged
def isconverged(previous, current):
    counter = 0
    #Checking number of thetas convergance
    checkcon = 0
    for x in current:
        #Printing the difference in current and previous theta
        print("Difference: ")
        print(x-previous[counter])
        if (x-previous[counter]) < threshold:
            checkcon = checkcon + 1
        counter = counter + 1
    print(checkcon)
    if checkcon == 9:
        #If all thetas are converged then return true and break the loop
        return True
    else:
        return False


#Function for calculating the gradient descent
def applygradientdescent(dataSet, startingthetas, alpha, maxIterations, dataLength):
    #Setting theta list equal to the initial thetas provided in parameters
    theta = [startingthetas[0],startingthetas[1],startingthetas[2],startingthetas[3],startingthetas[4],startingthetas[5],
             startingthetas[6],startingthetas[7],startingthetas[8],startingthetas[9]]
    #Setting old thetas to zero initially
    oldTheta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    i = 0
    #Break the loop when the number of iterations reach maximum number of iterations
    while i <= maxIterations:
        #Check if the model has converged
        if (isconverged(oldTheta, theta)):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Converged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            #Break the loop if model is converged
            break

        #Storing values of theta in oldTheta list
        oldTheta[0] = theta[0]
        oldTheta[1] = theta[1]
        oldTheta[2] = theta[2]
        oldTheta[3] = theta[3]
        oldTheta[4] = theta[4]
        oldTheta[5] = theta[5]
        oldTheta[6] = theta[6]
        oldTheta[7] = theta[7]
        oldTheta[8] = theta[8]
        oldTheta[9] = theta[9]
        j = 0

        #Initializing the cost function values to be zero initially
        sumcosttheta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        #Iterating for cost function against all thetas
        while j < dataLength:
            #Calculating hypothesis
            hypothesis = calculatehypothesis(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6],theta[7],
                                             theta[8], theta[9], dataSet[j][0], dataSet[j][1], dataSet[j][2], dataSet[j][3],
                                             dataSet[j][4], dataSet[j][5], dataSet[j][6], dataSet[j][7], dataSet[j][8])

            #Calculating cost function based upon values of hypothesis
            sumcosttheta[0] = sumcosttheta[0] + (hypothesis-dataSet[j][9]) * 0
            sumcosttheta[1] = sumcosttheta[1] + (hypothesis - dataSet[j][9]) * dataSet[j][0]
            sumcosttheta[2] = sumcosttheta[2] + (hypothesis - dataSet[j][9]) * dataSet[j][1]
            sumcosttheta[3] = sumcosttheta[3] + (hypothesis - dataSet[j][9]) * dataSet[j][2]
            sumcosttheta[4] = sumcosttheta[4] + (hypothesis - dataSet[j][9]) * dataSet[j][3]
            sumcosttheta[5] = sumcosttheta[5] + (hypothesis - dataSet[j][9]) * dataSet[j][4]
            sumcosttheta[6] = sumcosttheta[6] + (hypothesis - dataSet[j][9]) * dataSet[j][5]
            sumcosttheta[7] = sumcosttheta[7] + (hypothesis - dataSet[j][9]) * dataSet[j][6]
            sumcosttheta[8] = sumcosttheta[8] + (hypothesis - dataSet[j][9]) * dataSet[j][7]
            sumcosttheta[9] = sumcosttheta[9] + (hypothesis - dataSet[j][9]) * dataSet[j][8]
            j = j + 1

        #Printing the final cost function error
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(sumcosttheta)

        #######################GRADIENT DESCENT APPLIED HERE############################

        #Storing values of thetas into a temporary variable temptheta after applying GRADIENT DESCENT
        temptheta0 = theta[0] - alpha * (1 / dataLength) * sumcosttheta[0]
        temptheta1 = theta[1] - alpha * (1 / dataLength) * sumcosttheta[1]
        temptheta2 = theta[2] - alpha * (1 / dataLength) * sumcosttheta[2]
        temptheta3 = theta[3] - alpha * (1 / dataLength) * sumcosttheta[3]
        temptheta4 = theta[4] - alpha * (1 / dataLength) * sumcosttheta[4]
        temptheta5 = theta[5] - alpha * (1 / dataLength) * sumcosttheta[5]
        temptheta6 = theta[6] - alpha * (1 / dataLength) * sumcosttheta[6]
        temptheta7 = theta[7] - alpha * (1 / dataLength) * sumcosttheta[7]
        temptheta8 = theta[8] - alpha * (1 / dataLength) * sumcosttheta[8]
        temptheta9 = theta[9] - alpha * (1 / dataLength) * sumcosttheta[9]

        #Updating values of all the thetas simultaneously after storing in temporary variables
        theta[0] = temptheta0
        theta[1] = temptheta1
        theta[2] = temptheta2
        theta[3] = temptheta3
        theta[4] = temptheta4
        theta[5] = temptheta5
        theta[6] = temptheta6
        theta[7] = temptheta7
        theta[8] = temptheta8
        theta[9] = temptheta9

        #Printing Iteration number and values of all the thetas
        print("=============ITERATION===============")
        print(i)
        print("theta0: ")
        print(theta[0])
        print("theta1: ")
        print(theta[1])
        print("theta2: ")
        print(theta[2])
        print("theta3: ")
        print(theta[3])
        print("theta4: ")
        print(theta[4])
        print("theta5: ")
        print(theta[5])
        print("theta6: ")
        print(theta[6])
        print("theta7: ")
        print(theta[7])
        print("theta8: ")
        print(theta[8])
        print("theta9: ")
        print(theta[9])
        print("=========================================")
        i = i + 1

    #Returning final list of thetas
    return theta

#Setting initial values of theta
startingthetas = [0, 0.1, 0.01, 0.01, 0.001, 0.001, 0.001, 0.0001, 0.1, 0.0001]

#Scaling the features and storing scaled dataset in dataSetNew
dataSetNew = featureScaling(dataSet, dataLength)

#Gradient Descent Started here..
print("=========================================STARTING GRADIENT DESCENT=========================================")

#Running gradient descent by calling its function and getting final values of thetas
finalThetas = applygradientdescent(dataSetNew, startingthetas, alpha, maxIterations, dataLength)

#Printing final values of thetas
print("============== FINAL THETAS ====================")
print(finalThetas)
import sys
import pandas as pd
from pandas import DataFrame
import math

#Function for getting prediction
def prediction(x1,x2,x3,x4,x5,x6,x7,x8,x9):

    #Scaling the features x1 to x9 provided in input
    #Method used for scaling: x = x-min(x) / max(x)-min(x)
    x1 = round((x1 - 1) / (16 - 1), 4)
    x2 = round((x2 - 1990) / (2018 - 1990), 4)
    x3 = round((x3 - 50) / (999000 - 50), 4)
    x4 = round((x4 - 0) / (1 - 0), 4)
    x5 = round((x5 - 1) / (97 - 1), 4)
    x6 = round((x6 - 1) / (15 - 1), 4)
    x7 = round(((x7 - 0) / (1 - 0)), 4)
    x8 = round(((x8 - 600) / (4000 - 600)), 4)
    x9 = round(((x9 - 1) / (5 - 1)), 4)

    #Running hypothesis on scales features
    predictedValue = hypothesis(x1,x2,x3,x4,x5,x6,x7,x8,x9)

    #Printing Predicted Value
    print("Predicted Price:")
    print(round(predictedValue,4))

#Function for calculating hypothesis
#Hypothesis: theta0*0 + theta1*x1 + theta2*x2 + theta3*x3 + theta4*x4 + theta5*x5 + theta6*x6 + theta7*x7 + theta8*x8 + theta9*x9
#The values of theta are obtained from training
def hypothesis(x1,x2,x3,x4,x5,x6,x7,x8,x9):
    return 0 + (0.007338044213715173*(x1)) + (0.888939706560183*(x2)) + ((-3.025943824732459)*(x3)) + (x4*(-0.23669276432675987)) + (x5 * (0.046636558115992224)) + (x6 * (-0.4845866756324971)) + ((0.493541554346598) * x7) + ((4.959639160844014) * (x8*0.785)) + (x9 * (1.060079459901228))

#Testing function by passing the values into it (actual values from data set as it is are to be passed which further will be scaled accordingly)
prediction(2,2014,66122,1,4,2,0,1800,2)

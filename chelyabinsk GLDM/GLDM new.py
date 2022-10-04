# %%
import os
import sys
import pathlib
from types import CellType
# GUI
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QSizePolicy, QWidget, QFormLayout, QVBoxLayout, QHBoxLayout, QStyle, QAction, QFileDialog
from PyQt5.QtWidgets import QCheckBox, QComboBox, QDateEdit, QLabel, QLineEdit, QPushButton, QRadioButton, QSlider, QSpinBox
from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget, QScrollArea
# Data Read
import pandas as pd
# Plot
import matplotlib.pyplot as plt
import numpy as np
# Timing
import time
#Testing
from statsmodels.stats.diagnostic import acorr_ljungbox
plt.style.use('fivethirtyeight')
import statistics
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
import seaborn as sns
from pandas import read_csv, concat
from pandas.plotting import autocorrelation_plot
from scipy.stats import shapiro
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
#multiprocessing
import collections
import time
import os
import multiprocessing
from pprint import pprint
import numpy
## Calculate Error
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score
import math
from scipy.stats import pearsonr
import statistics

class GLDM():
    # %% Initialize variables
    def __init__(self, filePath=None):
        super(GLDM,self).__init__()
        self.Y = [] # target values
        self.size = 0
        self.dataRead = self.DataInput(filePath)
        if self.dataRead:
            self.n = 5 # Number of the summands
            self.m=self.size # Implementation lengths
            self.E=0
            self.D=0
            self.minFH=0 # reasonable forecasting horizon
            self.St=0 # St - start point, Et=St+T-1 - end point of the forecasting interval
            self.SZ=0
            self.GLDMEstimator_Time = 0

            # %% GForming
            self.G = [0 for i in range(6)]
            self.G[1] = lambda x1,x2 : x1
            self.G[2] = lambda x1,x2 : x2
            self.G[3] = lambda x1,x2 : x1*x1
            self.G[4] = lambda x1,x2 : x2*x2
            self.G[5] = lambda x1,x2 : x1*x2

            # %% MemAlloc
            rows, cols = (self.n+1, self.n+self.n+2)
            self.SST = [[0 for i in range(cols)] for j in range(rows)] # Matrix for J-G transforming 
            rows, cols = (self.m+2, self.n+1)
            self.P1 = [[0 for i in range(cols)] for j in range(rows)] # used for P calculation
            rows, cols = (self.m+2, self.m+2)
            self.P = [[0 for i in range(cols)] for j in range(rows)] # Projection matrix
            self.Prgrad = [0 for i in range(self.m+2)] # Projection of the gradient
            self.w = [0 for i in range(self.m+2)] # WLDM weights
            self.p = [1. for i in range(self.m+2)] # GLDM weights
            self.r = [0 for i in range(self.n+1)] # Ordinal numbers of the basic equations
            self.a = [0 for i in range(self.n+1)] # Identifyed parameters
            self.a1 = [0 for i in range(self.n+1)] # Identifyed parameters
            self.z = [0 for i in range(self.m+2)] # WLDM approximation errors (The difference between actual and modelled values) 
            self.FH = [0 for i in range(self.m+2)] # Reasonable Forecasting Horizons
        
            self.residuals = []
    def DataInput(self, filePath=None):
        result = False
        if filePath is None or filePath == '':
            filePath = "Data.txt"
        fileExtension = pathlib.Path(filePath).suffix
        if os.path.exists(filePath):
            if  fileExtension == '.txt':
                file = open(filePath,"r")
                # self.size = int(file.readline().split(":")[1].strip())
                # print(self.size)
                self.Y.append(0.0)
                while True:
                    line = file.readline().strip()
                    if line == "":
                        break
                    if line != "0":
                        self.Y.append(float(line))
                self.Y.append(0.0)
                file.close()
            elif fileExtension == '.csv':
                data = pd.read_csv(filePath, header=None, index_col=None)
                data = data[data[0] != 0]
                self.Y = [0] + data[0].values.tolist() + [0]
            elif fileExtension == '.xlsx':
                data = pd.read_excel(filePath, header=None, index_col=None)
                data = data[data[0] != 0]
                self.Y = [0] + data[0].values.tolist() + [0]
            self.size = len(self.Y)-2
            return True
        else:
            print("File does not exist")
            return False

    # %% SSTForming
    # Forming of projecting matrix SST with adjoint matrix
    # SST: Matrix for J-G transforming
    def SSTForming(self):
        for i in range(1,self.n+1):
            for j in range(1,self.n+1):
                self.SST[i][j]=0.
                for t in range(3,self.m+1):
                    A1=self.G[i](self.Y[t-2],self.Y[t-1])
                    A2=self.G[j](self.Y[t-2],self.Y[t-1])
                    self.SST[i][j]+=A1*A2
            # Adding the adjoint matrix for Jordan-Gauss algorithm
            for j in range(1,self.n+1):
                self.SST[i][self.n+j]=0
            self.SST[i][self.n+i]=1

    # %% JGTransforming Jordan-Gauss transforming
    # Jordan-Gauss transforming
    def JGTransforming(self, nn):
        for N in range(1,nn+1):
            # Find Lead Row
            mm=N # mm to find the lead row number
            Mi=0
            M=abs(self.SST[N][N])
            for i in range(N+1,nn+1):
                Mi=self.SST[i][N]
                if (abs(Mi>M)):
                    mm=i
                    M=Mi
            # Swapping of current N-th and lead mm-th rows
            Temp=0
            for K in range(1,2*nn+1):
                Temp=self.SST[N][K]
                self.SST[N][K]=self.SST[mm][K]
                self.SST[mm][K]=Temp

            # Normalise of the current row
            R=self.SST[N][N]
            for L in range(N,2*nn+1):
                try:
                    self.SST[N][L] /= R # ZeroDivisionError: float division by zero
                except ZeroDivisionError:
                    self.SST[N][L] = 1

            # Orthogonalize the Current Collumn
            for K in range(1,N):
                R=self.SST[K][N]
                for L in range(N,2*self.n+1):
                    self.SST[K][L] -= self.SST[N][L]*R

            for K in range(N+1,self.n+1):
                R=self.SST[K][N]
                for L in range(N,2*nn+1):
                    self.SST[K][L] -= self.SST[N][L]*R

    # %% GLDMEstimator
    # Forming of Matrix P1
    def P1Forming(self):
        for t in range(3,self.m+1):
            for j in range(1,self.n+1):
                self.P1[t][j]=0
                for k in range(1,self.n+1):
                    A1=self.G[k](self.Y[t-2],self.Y[t-1])
                    self.P1[t][j]+=A1*self.SST[k][self.n+j]

        print('\n' + "Matrix P1[3:m][1:n]" + '\n')
        for t in range(3,self.m+1):
            print('\n' + str(t) +'\t', end='')
            for j in range(1,self.n+1):
                print(str(self.P1[t][j]) + '\t', end='')

    # %%
    # Forming the Projecting Matrix P[3:m][3:m] 
    # PForming
    def PForming(self):    
        for t1 in range(3,self.m+1):
            for t2 in range(3,self.m+1):
                self.P[t1][t2]=0
                for j in range(1,self.n+1):
                    A1=self.G[j](self.Y[t2-2],self.Y[t2-1])
                    self.P[t1][t2]-=A1*self.P1[t1][j]
            self.P[t1][t1]+=1.

        print('\n' + "Matrix P[3:m][3:m]" + '\n', end='')
        for i in range(3,self.m+1):
            print('\n' + str(i) +'\t', end='')
            for j in range(3,self.m+1):
                print( str(self.P[i][j])  + '\t', end='')
        

    # %%
    # PrGradForming
    def PrGradForming(self):
        for i in range(3,self.m+1):
            self.Prgrad[i]=0.
            for j in range(3,self.m+1):
                self.Prgrad[i]+=self.P[i][j]*self.Y[j]
                # gradient projection is found

        print('\n' + "i   Y[i]   Prgrad[i]    p[i]  " + '\n', end='')
        for i in range(3,self.m+1):
            print('\n' + str(i) +'\t' + str(self.Y[i])  +'\t' + str(self.Prgrad[i]) + '\t' + str(self.p[i]), end='')

    # %%
    # The function finds a solution to the dual problem w [t] and the number of active constraints
    # DualWLDMSolution()
    def DualWLDMSolution(self):
        LARGE=sys.maxsize
        Al=LARGE
        Alc=0
        for t in range(3,self.m+1):
            self.w[t]=0
        C=0 # the number of active constraints
        while C < self.m-self.n-2: # Finding the Length of Moving along Prgrad
            Al=LARGE # Al - offset length
            for t in range(3, self.m+1): # For each coordinate w [t], we narrow down the possible values of Al
                # p[t] - the given weights
                # w[t] - the variables of the dual problem
                if (abs(self.w[t])==self.p[t]):
                    # w [t] takes a boundary value, we transfer it to the status of fixed ones
                    continue
                else:
                    # In the occasional case of the input, the variable time is measured along the range of the gradient with the fixed t
                    if (self.Prgrad[t]>0):
                        Alc=(self.p[t]-self.w[t])/self.Prgrad[t] 		
                    elif (self.Prgrad[t]<0):
                        Alc=(-self.p[t]-self.w[t])/self.Prgrad[t] 				
                    if(Alc<Al):
                        Al=Alc # offset length
            # After the cooling of the Al cycle, it contains the maximum offset
            # Length of moving along Prgrad is equal to Al
            for jj in range(3,self.m+1):
                if (abs(self.w[jj])!=self.p[jj]):
                    self.w[jj]+=Al*self.Prgrad[jj] # For points that are not needed on the boundary, 
                    # we make a step of length Al in the direction of the graient
                    if(abs(self.w[jj])==self.p[jj]):
                        C=C+1 # When the limit is lowered by 1 the number of the active constraints
                            
    # %%
    # PrimalWLDMSolution
    def PrimalWLDMSolution(self):
        ri=0 # ri equal to number of the basic equations
        for t in range(3,self.m+1):
            if (abs(self.w[t])!=self.p[t]):
                ri=ri+1 
                self.r[ri]=t
                
        for l in range(1, ri+1): # Formig of the Equations 
            for i in range(1,ri+1):
                A1=self.G[i](self.Y[self.r[l]-1],self.Y[self.r[l]-2])
                self.SST[l][i]=A1
            self.SST[l][ri+1]=self.Y[self.r[l]]
        self.JGTransforming( ri)
        for i in range(1,ri+1):
            self.a[i]=self.SST[i][ri+1]
            self.z[self.r[i]]=0

        print('\n', end='')
        for i in range(1,self.n+1):
            print("a[" + str(i) + "]=" + str(self.a[i]) + '\t', end='')
        print('\n', end='')


    # %%
    def GLDMEstimator(self):
        start = time.time()
        self.SSTForming()
        self.JGTransforming(self.n)

        self.P1Forming()
        self.PForming()
        self.PrGradForming()

        d = 1
        while d>0:
            for ii in range(1,self.n+1):
                self.a1[ii]=self.a[ii]
            for ii in range(1,self.m+1):
                self.p[ii]=1./(1.+self.z[ii]*self.z[ii])
            for ii in range(1,self.m+1):
                self.w[ii]=0.
            self.DualWLDMSolution() # Solution of dual problem
            self.PrimalWLDMSolution() # Solution of primal problem
            Z=self.z[1]=self.z[2]=0. # Defining the residuals
            # Z be the loss function
            # z be the vctor of residuals
            Zs = []
            for t in range(3,self.m+1):
                self.z[t]=self.Y[t]
                for i in range(1,self.n+1):
                    A1=self.G[i](self.Y[t-1],self.Y[t-2])
                    self.z[t]-=self.a[i]*A1
                Z+=abs(self.z[t])  #data in model
                Zs.append((self.z[t]))
            d=abs(self.a[1]-self.a1[1])
            for i in range(2,self.n+1):
                if(d<abs(self.a[i]-self.a1[i])):
                    d=abs(self.a[i]-self.a1[i])  

        self.GLDMEstimator_Time = time.time() - start

        self.residuals = Zs
        # Ljungbox
        g = open("Error.txt","w")
        LjungBox=acorr_ljungbox(Zs, lags=[len(Zs)-1], return_df=True) 
        if LjungBox.lb_pvalue[len(Zs)-1]<=0.05:
            g.write('\n'+f'Rejection H0 and The residuals are independently distributed (Serial Uncorrelated)')
        else:
            g.write('\n'+f"Accepted H0 and The residuals are not independently distributed; they exhibit serial correlation.")
        shapirotest=shapiro(Zs)
        if shapirotest.pvalue<=0.05:
            g.write('\n'+f'Rejection H0 that Data follows Normal Distribution')  
        else:
            g.write('\n'+f"Accepted H0 that signals does not follows Normal Distribution")
        mean=statistics.mean(Zs)
        # summary statistics
        residuals = Zs
        residuals = pd.DataFrame(residuals)
        print(residuals.describe())
        g.write('\n'"Mean"'\n'+str(mean))
        g.close()


        residuals = Zs
        residuals = pd.DataFrame(residuals)

        # autocorrelation_plot
        fig = plt.figure(figsize=(20, 10))
        plt.title('autocorrelation_plot')
        autocorrelation_plot(residuals)
        plt.show()
        plt.savefig('autocorrelation_plot.png')

        # qqplot
        # create lagged dataset
        #from pandas import DataFrame
        #values = DataFrame(residuals.values)
        #dataframe = concat([values.shift(1), values], axis=1)
        #dataframe.columns = ['t-1', 't+1']
        #fig = plt.figure(figsize=(20, 10))
        #plt.title('qqplot')
        #residuals = numpy.array(residuals)
        #qqplot(residuals)
        #plt.show()
        #plt.savefig('qqplot.png')

        # histogram plot
        residuals = Zs
        residuals = pd.DataFrame(residuals)
        residuals.hist()
        plt.title('Histogram Plot of Residual Errors')
        plt.show()
        plt.savefig('Histogram Plot of Residual Errors.png')
    
        # Density Plot of Residual Errors
        residuals = Zs
        residuals = pd.DataFrame(residuals)
        residuals.plot(kind='kde')
        plt.title('Density Plot of Residual Errors')
        plt.show()
        plt.savefig('Density Plot of Residual Errors.png')


        # plot residuals (Line Plot of Residual Errors)
        residuals.plot()
        plt.title('Line Plot of Residual Errors')
        plt.show()
        plt.savefig('Line Plot of Residual Errors.png')

        g = open("Description.txt","w")
        g.write('\n'+f'Residual Autocorrelation Plot')
        g.write('\n'+f'Autocorrelation calculates the strength of the relationship between an observation and observations at prior time steps.')
        g.write('\n'+f"We can calculate the autocorrelation of the residual error time series and plot the results. \n This is called an autocorrelation plot.")
        g.write('\n'+f'We would not expect there to be any correlation between the residuals.\n This would be shown by autocorrelation scores being below the threshold of significance \n (dashed and dotted horizontal lines on the plot).')
        g.write('\n'+f'visualizing the autocorrelation for the residual errors. \n The x-axis shows the lag and the y-axis shows the correlation between an observation and the lag variable, \n where correlation values are between -1 and 1 for negative and positive correlations respectively.')
        g.write('\n'+f'Residual Histogram and Density Plots')
        g.write('\n'+f'Plots can be used to better understand the distribution of errors beyond summary statistics.\n We would expect the forecast errors to be normally distributed around a zero mean. \n Plots can help discover skews in this distribution. \n We can use both histograms and density plots to better understand the distribution of residual errors.')
        mean=statistics.mean(Zs)
        # summary statistics
        residuals = Zs
        residuals = pd.DataFrame(residuals)
        print(residuals.describe())
        g.write('\n'"Mean"'\n'+str(mean))
        g.write('\n'"Mean"'\n'+str(residuals.describe()))
        g.close()
        return Z
    # %%
    # Calculation of the average prediction errors
    def ForecastingEst(self):
        rows, cols = (self.m+2, self.m+2)
        self.PY = [[0 for i in range(cols)] for j in range(rows)] # PY[i][t] is forward-looking forecast Y[i+t] 

        self.St = 0
        # Et
        while True:
            self.St=self.St+1
            self.PY[self.St][0]=self.Y[self.St]
            self.PY[self.St][1]=self.Y[self.St+1]
            t=1 # // forecasting horizon (time horizon)
            while True:
                try:
                    t=t+1
                    self.PY[self.St][t]=0
                    for j in range(1,self.n+1):
                        A1=self.G[j](self.PY[self.St][t-1],self.PY[self.St][t-2])
                        # PY[St][2] = G[j](PY[St][1],PY[St][0])
                        self.PY[self.St][t]+=self.a[j]*A1
                    if self.St >= len(self.PY) or (self.St+t) >= len(self.Y) or t >= self.size:
                        break
                    print(f't={t} St={self.St} PY[St][t]={self.PY[self.St][t]} St+t={self.St+t} Y[St+t]={self.Y[self.St+t]}  diff={abs(self.PY[self.St][t]-self.Y[self.St+t])} SZ={self.SZ}')
                    if not (abs(self.PY[self.St][t]-self.Y[self.St+t]) <= self.SZ):
                        break
                except:
                    ...
            self.FH[self.St]=t # FH[St] - reliable
            print( '\n' + " St=" + str(self.St) + "  FH=[St]= " + str(t) + '\n')
            if t == 2:
                break
        # Now St equal to number of the used fragments

        # find minimal FH[t] for all t<St
        self.minFH=self.FH[self.St]
        for t in range(3,self.St):
            if (self.FH[t]<self.minFH):
                self.minFH=self.FH[t]
        # Now minFH is equal to reasonable forecasting horizon 


        forecastList = []
        for st in range(2,self.St):
            # self.St + self.FH[self.St] < self.m
            forecasted = self.Y[1:st] + self.PY[st][1:-st]
            if 0 not in forecasted:
                # plt.plot(x, forecasted, label='Predicted data')
                forecastList.append(forecasted)
        averageForecast = np.array(forecastList).sum(axis=0)/len(forecastList)
        self.St=1
        self.minFH=self.m

        self.E=self.D=0
        for t in range(3,self.minFH):
            self.D+=abs(self.Y[t+self.St]-averageForecast[t]) # The summ of absolute errors of the prediction Y[i] by the values Y[i-T-1] and Y[i-T]
            self.E+=(self.Y[t+self.St]-averageForecast[t]) # The summ of errors of the prediction Y[i] by the values Y[i-T-1] and Y[i-T]
        # The average errors of the prediction for time horizon minFH
        self.D/=self.minFH
        self.E/=self.minFH

        return averageForecast


    # %%
    # Calculation of the average prediction errors
    def ForecastingFuture(self, days, valu0, valu1):
        futureDays = [0 for i in range(days + 2)]
        futurePredictions = [0 for i in range(days)]
        
        futureDays[0]=valu0
        futureDays[1]=valu1
        # Et
        t=1 # // forecasting horizon (time horizon)
        for day in range(days):
            t=t+1
            futureDays[t]=0
            for j in range(1,self.n+1):
                A1=self.G[j](futureDays[t-1],futureDays[t-2])
                futureDays[t]+=self.a[j]*A1
            futurePredictions[day] = futureDays[t]
            print(" t=" + str(t) + "  futureDays[t]= " + str(futureDays[t]))
        return futurePredictions

class Main(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        widget = QWidget()
        self.gldm = GLDM()

        lblDataType = QLabel('Type of Data')
        dataType = QComboBox()
        dataType.addItem('Daily')
        dataType.addItem('Weekly')
        dataType.addItem('Monthly')
        dataType.addItem('Quarterly')
        dataType.addItem('Annually')

        lblName = QLabel('Name')
        Name = QLineEdit()

        lblDays = QLabel('Prediction Days')
        days = QSpinBox()
        days.setMaximum(100000)

        lblFile = QLabel('Data Input')
        filePath = QLabel()
        btnFile = QPushButton('Pick')
        def ChooseFile():
            filedialog = QFileDialog()
            filedialog.setDefaultSuffix("txt")
            filedialog.setNameFilter("Text Files (*.txt);CSV Files (*.csv);Excel Files (*.xlsx);")
            filedialog.setAcceptMode(QFileDialog.AcceptOpen)
            filePath.setText(filedialog.getOpenFileName()[0])
            self.gldm = GLDM(filePath.text())
        btnFile.clicked.connect(ChooseFile)


        lblLegendLocation = QLabel('Legend Location')
        legendlocation = QComboBox()
        legendlocation.addItem('upper right')
        legendlocation.addItem('upper left')
        legendlocation.addItem('lower right')
        legendlocation.addItem('lower left')

        fileBox = QHBoxLayout()
        fileBox.addWidget(filePath)
        fileBox.addWidget(btnFile)

        btnSubmit = QPushButton('Submit')
        
        fbox = QFormLayout()
        fbox.addRow(lblFile, fileBox)
        fbox.addRow(btnSubmit)
        fbox.addRow(lblLegendLocation,legendlocation)
        fbox.addRow(lblName,Name)
        fbox.addRow(lblDataType,dataType)
        fbox.addRow(lblDays,days)
        scroll = QScrollArea()
        #Scroll Area Content
        scrollable = QWidget()
        verticalScroll = QVBoxLayout()
        scrollable.setLayout(verticalScroll)
        #Scroll Area Properties
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(scrollable)
        fbox.addRow(scroll)
        def format2digits(value):
            # return format(value, ".2f")
            # return str(round(value, 2))
            return str(value)
        def submitClicked():
            if self.gldm.dataRead:
                # Solution
                self.gldm.SSTForming() # Forming of projecting matrix # Data Preparation
                self.gldm.JGTransforming(self.gldm.n) # Jordan-Gauss transforming # Data Preparation
                self.gldm.SZ=self.gldm.GLDMEstimator() # Procedure of estimating using Generalized Least Deviation Method # Training
                averageForecast = self.gldm.ForecastingEst() # Calculation of the average prediction errors # Testing
                forecastDays = days.value() # Forecasting
                predictionX, forecastX = np.linspace(1, self.gldm.m, self.gldm.m), np.linspace(self.gldm.m+1, self.gldm.m+forecastDays, forecastDays)
                forecastY = self.gldm.ForecastingFuture(forecastDays, averageForecast[self.gldm.m-2], averageForecast[self.gldm.m-1]) # Forecasting               
                plt.figure(figsize=(15,10))
                plt.plot(predictionX, averageForecast, label='Model', linewidth=1, color='black')
                plt.plot(predictionX, self.gldm.Y[1:self.gldm.m+1], label='Actual data', linewidth=1, color='b')
                plt.plot(forecastX, forecastY, label='Forecasted data', linewidth=1, color='green')
                plt.legend(loc=legendlocation.currentText())
                plt.ylim(0,max(self.gldm.Y[1:self.gldm.m+1])+self.gldm.m)
                plt.xlabel(f'Time ({dataType.currentText()})')
                plt.ylabel(Name.text())
                plt.title(Name.text())
                plt.grid()
                plt.show()
                plt.savefig('Model.png')

                g = open("Out.txt","w")
                g.write("Optimal factors : ")
                for i in range(1,self.gldm.n+1):
                    g.write('\n'+"a["+str(i)+"]="+str(self.gldm.a[i]))
                g.write('\n'+'\n'+"Optimal value of Loss function : "+str(self.gldm.SZ)+'\n')
                g.write('\n'+"reasonable forecasting horizon = "+str(self.gldm.minFH))
                g.write('\n'+"Average prediction errors: E="+str(self.gldm.E)+"; D="+str(self.gldm.D))
                g.write('\n'+f"GLDMEstimator Time Consumed {self.gldm.GLDMEstimator_Time:.3f} seconds")
                for t in range(forecastDays):
                    g.write('\n'+" t=" + str(t+1) + "  futureDays[t]= " + format2digits(forecastY[t]))
                g.close()               
                verticalScroll.addWidget(QLabel("Optimal factors : "))
                for i in range(1,self.gldm.n+1):
                    verticalScroll.addWidget(QLabel("a["+str(i)+"]="+format2digits(self.gldm.a[i])))
                verticalScroll.addWidget(QLabel("Optimal value of Loss function : "+format2digits(self.gldm.SZ)))
                verticalScroll.addWidget(QLabel("reasonable forecasting horizon = "+format2digits(self.gldm.minFH)))
                verticalScroll.addWidget(QLabel("Average prediction errors: E="+format2digits(self.gldm.E)+"; D="+format2digits(self.gldm.D)))
                verticalScroll.addWidget(QLabel(f"GLDMEstimator Time Consumed {self.gldm.GLDMEstimator_Time:.3f} seconds"))
                for t in range(forecastDays):
                    verticalScroll.addWidget(QLabel(" t=" + str(t+1) + "  futureDays[t]= " + format2digits(forecastY[t])))

                trainY= self.gldm.Y[1:-1] # self.gldm.Y
                trainPredict = averageForecast
                trainY= self.gldm.Y[1:-1] # self.gldm.Y
                trainPredict = averageForecast

                def gprint(print_statement):
                    # print(print_statement)
                    g = open("Error Matrix.txt","a")
                    g.write(print_statement+'\n')
                    g.close()
                #Mean Squared Error
                if os.path.isfile('Error.txt'):
                    os.remove('Error.txt')
                trainScore_MSE = mean_squared_error(trainY, trainPredict)
                gprint('Train Score: %.2f MSE' % (trainScore_MSE))
                # calculate root mean squared error
                trainScore_RMSE = math.sqrt(mean_squared_error(trainY, trainPredict))
                gprint('Train Score: %.2f RMSE' % (trainScore_RMSE))
                #  Mean Absolute Error
                trainScore_MAE = mean_absolute_error(trainY, trainPredict)
                gprint('Train Score: %.2f MAE' % (trainScore_MAE))
                #  Mean Absolute Percentage Error
                trainScore_MAPE = mean_absolute_percentage_error(trainY, trainPredict)
                gprint('Train Score: %.2f MAPE' % (trainScore_MAPE))
                #  Coefficient of Determination-R2 score
                trainScore_r2 = r2_score(trainY, trainPredict)
                gprint('Train Score: %.2f r2' % (trainScore_r2))
                # RRMSE relative root mean square error (RRMSE) is calculated by dividing the RMSE by the mean observed data 
                RRMSE_train = trainScore_RMSE/statistics.mean(trainY)
                gprint('Train Score: %.2f RRMSE' % (RRMSE_train))
                # calculate Pearson's correlation
                trainScore_correlation = pearsonr(trainY, trainPredict)
                gprint('Train Score: %.2f correlation ' % (trainScore_correlation[0]))
                # calculate Mean Bias Error(MBE) for a set of actual and test prediction
                MBE_train = np.mean(trainPredict - trainY) #here we calculate MBE
                gprint('Train Score: %.2f Mean Bias Error ' % (MBE_train))

        btnSubmit.clicked.connect(submitClicked)
        
        self.setGeometry(100, 60, 60, 100)
        self.move(QDesktopWidget().availableGeometry().center() - self.frameGeometry().center())

        widget.setLayout(fbox)
        self.setCentralWidget(widget)
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    app.exec_()



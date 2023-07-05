from sklearn.metrics import classification_report
import numpy as np
import math
import time
from sklearn.neighbors import KNeighborsClassifier
import os
from shutil import copyfile
from sklearn import svm

def GetMissedClassified(testClasses,predictions):
    A_missed=0
    A_total=0
    B_missed=0
    B_total=0
    C_missed=0
    C_total=0
    D_missed=0
    D_total=0
    
    for i in range(len(testClasses)):
        if testClasses[i] == 1:
            A_total = A_total + 1
            if predictions[i] != 1:
                A_missed = A_missed + 1
                
        elif testClasses[i] == 2:
            B_total = B_total + 1
            if predictions[i] != 2:
                B_missed = B_missed + 1
        
        elif testClasses[i] == 3:
            C_total = C_total + 1
            if predictions[i] != 3:
                C_missed = C_missed + 1
                
        elif testClasses[i] == 4:
            D_total = D_total + 1
            if predictions[i] != 4:
                D_missed = D_missed + 1
               
                    
    A_accuracy= (1-(A_missed/A_total))*100.0
    B_accuracy= (1-(B_missed/B_total))*100.0
    C_accuracy= (1-(C_missed/C_total))*100.0
    D_accuracy= (1-(D_missed/D_total))*100.0
    #'\t\t' + 'A: ' + str(A_accuracy) + '\t' + 'B: ' + str(B_accuracy)
    return (A_accuracy, B_accuracy, C_accuracy, D_accuracy)

def FiveFoldCrossValidation(dataA, dataB, dataC, dataD):
  
  #the 5 cross validation, 80% of the data used for trainning and 20% will be used for the testing.
  #take first 20% of the data and mark them as test NC[s, 2] = 1 mean test and 0 means trainning!!!
  num = 5;
  percent = 100 / num;
  testSampleCountA = int(((len(dataA) * percent) / 100))#take 80% of the data for training *2 because we have 2 classes (C, NC)
  testSampleCountB = int(((len(dataB) * percent) / 100))#take 80% of the data for training *2 because we have 2 classes (C, NC)
  testSampleCountC = int(((len(dataC) * percent) / 100))#take 80% of the data for training *2 because we have 2 classes (C, NC)
  testSampleCountD = int(((len(dataD) * percent) / 100))#take 80% of the data for training *2 because we have 2 classes (C, NC)
  
  accuracyTotal=0
  A_total=0
  B_total=0
  C_total=0
  D_total=0
  for cross in range(num):
        trainData=np.empty((0,9*58+1), np.float64)
        testData=np.empty((0,9*58+1), np.float64)
        testFiles=[]
        for i in range(len(dataA)):
          dataA[i][9*58]=0
    
        for i in range(len(dataB)):
          dataB[i][9*58]=0
          
        for i in range(len(dataC)):
          dataC[i][9*58]=0
    
        for i in range(len(dataD)):
          dataD[i][9*58]=0
        #now do the trainning and testing
        #load the test data
        i = (cross * testSampleCountA)
        while(i < ((cross + 1) * testSampleCountA)):
          dataA[i][9*58]=1
          i+=1
      
        i = (cross * testSampleCountB)
        while(i < ((cross + 1) * testSampleCountB)):
         dataB[i][9*58]=1
         i+=1
        
        i = (cross * testSampleCountC)
        while(i < ((cross + 1) * testSampleCountC)):
          dataC[i][9*58]=1
          i+=1
      
        i = (cross * testSampleCountD)
        while(i < ((cross + 1) * testSampleCountD)):
         dataD[i][9*58]=1
         i+=1
         
      #load the trainning data A    
        for i in range(len(dataA)):
          #tm=[0,0,0]
          tm=np.empty((1,9*58+1), np.float64)
          for k in range(9*58):
              if math.isnan(dataA[i][k]):
                  tm[0][k]=0
              else:
                  tm[0][k]=dataA[i][k]
                  
          tm[0][9*58]=1#class A =0
          if(dataA[i][9*58] == 0):
            trainData=np.append(trainData,tm, axis=0)
          else:
            testData=np.append(testData,tm, axis=0)
      
      #load the trainning data B    
        for i in range(len(dataB)):
          tm=np.empty((1, 9*58+1), np.float64)
          for k in range(9*58):
              if math.isnan(dataB[i][k]):
                  tm[0][k]=0
              else:
                  tm[0][k]=dataB[i][k]
                  
          tm[0][9*58]=2#class B =1
          if(dataB[i][9*58] == 0):      
            trainData=np.append(trainData,tm, axis=0)
          else:
            testData=np.append(testData,tm, axis=0)
            
      #load the trainning data C    
        for i in range(len(dataC)):
          tm=np.empty((1, 9*58+1), np.float64)
          for k in range(9*58):
              if math.isnan(dataC[i][k]):
                  tm[0][k]=0
              else:
                  tm[0][k]=dataC[i][k]
                  
          tm[0][9*58]=3#class B =1
          if(dataC[i][9*58] == 0):      
            trainData=np.append(trainData,tm, axis=0)
          else:
            testData=np.append(testData,tm, axis=0)
            
      #load the trainning data C    
        for i in range(len(dataD)):
          tm=np.empty((1, 9*58+1), np.float64)
          for k in range(9*58):
              if math.isnan(dataD[i][k]):
                  tm[0][k]=0
              else:
                  tm[0][k]=dataD[i][k]
                  
          tm[0][9*58]=4#class B =1
          if(dataD[i][9*58] == 0):      
            trainData=np.append(trainData,tm, axis=0)
          else:
            testData=np.append(testData,tm, axis=0)
            
        testClasses = testData[:,9*58]
        trainClasses = trainData[:,9*58]
        testData= np.delete(testData,9*58,1)
        trainData= np.delete(trainData,9*58,1)
        
        import warnings #to ignore warnnings coming out!
        warnings.filterwarnings('ignore')
        #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        #kernel='linear'
        clf = svm.SVC(gamma='scale', kernel='poly')
        clf.fit(trainData, trainClasses)
        predictions =clf.predict(testData)
     
        report=classification_report(testClasses,predictions)
        acu=report.split('\n')[7].split(' ')[31]
        accuracyTotal= accuracyTotal + float(acu)
        #print('Fold:'+str(cross+1)+': ' + acu)
        A_accu, B_accu, C_accu, D_accu =GetMissedClassified(testClasses,predictions)
        A_total = A_total + A_accu
        B_total = B_total + B_accu
        C_total = C_total + C_accu
        D_total = D_total + D_accu
    #str(margin)+":\t"+
  print(str(round((accuracyTotal*100)/5.0, 3))+"\t"+str(round(A_total/5.0, 2)) + "\t"+
        str(round(B_total/5.0, 2))
        + "\t"+str(round(C_total/5.0, 2))
        + "\t"+str(round(D_total/5.0, 2)))
  # print('A_Total: ' + str(round(A_total/5.0, 2)))
  # print('B_Total: ' + str(round(B_total/5.0, 2)))
  #  print (knn(trainData, testData, 10, 4))
        
sectorsNr=9
for margin in range(1, 25):   
    #load the 100 point distances into two categories!
    path="E:\\MyData\\DataSet\\ThyroidDS\\Thyroid Cancer Signs All Cases\\LBP\\LBP_MarginSectors\\HistogramAddingPixelIntensity\\AllSectors\\"+str(sectorsNr) +"Sectors\\58\\SizeNormEachSector\\lbp_margin_"+str(margin)+".txt"
    distfile = open(path, "r")
    
    dataA=np.empty((0,9*58+1), np.float64)
    dataB=np.empty((0,9*58+1), np.float64)
    dataC=np.empty((0,9*58+1), np.float64)
    dataD=np.empty((0,9*58+1), np.float64)
    
    while True:
      s=distfile.readline()
      if s == '':
        break
      
      sp=s.split('\t')
      fl = sp[0]
      percOrg= float(sp[1])
      tmp = np.empty((1, 9*58+1), np.float64)
      pts=sp[2].split(',')
      
      for i in range(len(pts)):
          tmp[0][i]= float(pts[i])
          
      if percOrg < 20:
          dataA = np.append(dataA,tmp, axis=0)
      elif percOrg < 35:
          dataB = np.append(dataB,tmp, axis=0)
      elif percOrg < 70:
          dataC = np.append(dataC,tmp, axis=0)
      else:
          dataD = np.append(dataD,tmp, axis=0)
    
    distfile.close()
    
    FiveFoldCrossValidation(dataA, dataB, dataC, dataD)
    
    
    
    
    

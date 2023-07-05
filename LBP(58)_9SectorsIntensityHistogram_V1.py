import os
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from numpy import linalg, array, sqrt, mean
import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib.path as mplPath
import sys


def GetULBP58():
    ulbp = np.zeros(shape=(58), dtype=np.ubyte);
    
    j=0
    bc = np.ubyte(0b00000000)
    ulbp[j] = bc
    
    j=1
    bc = np.ubyte(0b00000001)
    for i in range(8):
        ulbp[j] = bc
        #print(bc)
        j= j+1
        #to rotate the 8 bit binary value
        bc = np.ubyte((bc << 1) | (bc >> 7))
        
    bc = np.ubyte(0b00000011)
    for i in range(8):
        ulbp[j] = bc
        #print(bc)
        j= j+1
        #to rotate the 8 bit binary value
        bc = np.ubyte((bc << 1) | (bc >> 7))
        
    bc = np.ubyte(0b00000111)
    for i in range(8):
        ulbp[j] = bc
        #print(bc)
        j= j+1
        #to rotate the 8 bit binary value
        bc = np.ubyte((bc << 1) | (bc >> 7))
    bc = np.ubyte(0b00001111)
    for i in range(8):
        ulbp[j] = bc
        #print(bc)
        j= j+1
        #to rotate the 8 bit binary value
        bc = np.ubyte((bc << 1) | (bc >> 7))
    
    bc = np.ubyte(0b00011111)
    for i in range(8):
        ulbp[j] = bc
        #print(bc)
        j= j+1
        #to rotate the 8 bit binary value
        bc = np.ubyte((bc << 1) | (bc >> 7))
    
    bc = np.ubyte(0b00111111)
    for i in range(8):
        ulbp[j] = bc
        #print(bc)
        j= j+1
        #to rotate the 8 bit binary value
        bc = np.ubyte((bc << 1) | (bc >> 7))
    
    bc = np.ubyte(0b01111111)
    for i in range(8):
        ulbp[j] = bc
        #print(bc)
        j= j+1
        #to rotate the 8 bit binary value
        bc = np.ubyte((bc << 1) | (bc >> 7))
        
    bc = np.ubyte(0b11111111)
    ulbp[j] = bc
    #print(bc)
    return ulbp


def DrawRibbonSectors_58_LBP(X, Y, margin, MyImg):
    img1 = ImageDraw.Draw(MyImg) 
    # coordinates of the barycenter
    # centre of mass!
    x_m = np.mean(X)
    y_m = np.mean(Y)
         
    # calculation of the reduced coordinates
    u = X - x_m
    v = Y - y_m
         
    Suv  = sum(u*v)
    Suu  = sum(u**2)
    Svv  = sum(v**2)
    Suuv = sum(u**2 * v)
    Suvv = sum(u * v**2)
    Suuu = sum(u**3)
    Svvv = sum(v**3)
    
    # Solving the linear system
    A = array([ [ Suu, Suv ], [Suv, Svv]])
    B = array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = linalg.solve(A, B)
    
    cCentreX = x_m + uc
    cCentreY = y_m + vc
    
    # Calcul des distances au centre (cCentreX, cCentreY)
    Ri_1     = sqrt((X-cCentreX)**2 + (Y-cCentreY)**2)
    R_1      = mean(Ri_1)
    residu_1 = sum((Ri_1-R_1)**2)
    
    #interpolate the ROI points to get smooth boarder curve!
    datax= np.append(X, X[0])
    datay= np.append(Y, Y[0])
    xx=np.linspace(0, len(datax)-1, len(datax)) 
    fx = interp1d(xx, datax, kind='cubic')
    fy = interp1d(xx, datay, kind='cubic')
    xnew = np.linspace(xx[0], xx[len(xx)-1], num=len(datax)*3, endpoint=True)
    #now we find points of certain distance from ROI points along the line 
    #coming from centre of circle 
    Xout = []
    Yout = []
    Xin = []
    Yin = []
    for i in range(0, len(xnew)): 
         clr = (255, 0, 0)
         #img1.point((fx(xnew[i]), fy(xnew[i])), fill=clr)
         # shape = [(fx(xnew[i])-1, fy(xnew[i])-1), (fx(xnew[i])+1, fy(xnew[i])+1)]
         # img1.ellipse(shape, fill=clr, outline=clr, width=0)
         # clr = (255, 255, 255)
         # shape = [(fx(xnew[i]), fy(xnew[i])), (cCentreX, cCentreY)]
         # img1.line(shape, fill=clr, width=1)
         #The gradient of the line between two points is:
         m=(fy(xnew[i])-cCentreY)/(fx(xnew[i])-cCentreX) 
         Bi=fy(xnew[i])-m*fx(xnew[i]) #Bi is y intercept!
         #circle equation (x, y) point intersection between the circle and the line
         #(x−x0)2+(y−y0)2=d2  (x0, y0) are circle centre, d is radius
         #https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
         dis=math.sqrt((fx(xnew[i])-cCentreX)**2 + (fy(xnew[i])-cCentreY)**2)
         dist = dis - margin
         Xin1 = cCentreX + (dist/math.sqrt(1+m**2))
         Yin1 = Bi + m*Xin1
         Xin2 = cCentreX - (dist/math.sqrt(1+m**2))
         Yin2 = Bi + m*Xin2
         #now for each ROI point we get two margin points, which we need to decide which one is the right one
         #we use the point which is closest to the actual ROI point
         dis1=math.sqrt((fx(xnew[i])-Xin1)**2 + (fy(xnew[i])-Yin1)**2)
         dis2=math.sqrt((fx(xnew[i])-Xin2)**2 + (fy(xnew[i])-Yin2)**2)
         
         clr = (0, 0, 255)
         if(dis1 < dis2):
             #shape = [(Xin1-1, Yin1-1), (Xin1+1, Yin1+1)]
             Xin.append(Xin1)
             Yin.append(Yin1)
         else:
             #shape = [(Xin2-1, Yin2-1), (Xin2+1, Yin2+1)]
             Xin.append(Xin2)
             Yin.append(Yin2)
             
         #img1.ellipse(shape, fill=clr, outline=clr, width=0)
         
         dist = dis + margin
         Xout1 = cCentreX + (dist/math.sqrt(1+m**2))
         Yout1 = Bi + m*Xout1
         Xout2 = cCentreX - (dist/math.sqrt(1+m**2))
         Yout2 = Bi + m*Xout2
         #now for each ROI point we get two margin points, which we need to decide which one is the right one
         #we use the point which is closest to the actual ROI point
         dis1=math.sqrt((fx(xnew[i])-Xout1)**2 + (fy(xnew[i])-Yout1)**2)
         dis2=math.sqrt((fx(xnew[i])-Xout2)**2 + (fy(xnew[i])-Yout2)**2)
         
         clr = (0, 0, 255)
         if(dis1 < dis2):
             #shape = [(Xout1-1, Yout1-1), (Xout1+1, Yout1+1)]
             Xout.append(Xout1)
             Yout.append(Yout1)
         else:
             #shape = [(Xout2-1, Yout2-1), (Xout2+1, Yout2+1)]
             Xout.append(Xout2)
             Yout.append(Yout2)
             
         #img1.ellipse(shape, fill=clr, outline=clr, width=0)
    
    #drawing a bounding box around the outer margin!
    #ading one pixel to each side of the bounding box for HOG be able to calculate
    #at bounding box edge pixels.
    # xmax=max(Xout)+1
    # xmin=min(Xout)-1
    # ymax=max(Yout)+1
    # ymin=min(Yout)-1
        
    # clr=(255,0,0)
    # shape = [(xmin, ymin), (xmax, ymin)]
    # img1.line(shape, fill=clr, width=1)
    # shape = [(xmax, ymin), (xmax, ymax)]
    # img1.line(shape, fill=clr, width=1)
    # shape = [(xmax, ymax), (xmin, ymax)]
    # img1.line(shape, fill=clr, width=1)
    # shape = [(xmin, ymax), (xmin, ymin)]
    # img1.line(shape, fill=clr, width=1)
    
    maxDist=0
    clr = (255, 0, 0)
    for i in range(0, len(Xout)): 
        # shape = [(Xout[i]-1, Yout[i]-1), (Xout[i]+1, Yout[i]+1)]
        # img1.ellipse(shape, fill=clr, outline=clr, width=0)
         
        dis=math.sqrt((Xout[i]-cCentreX)**2 + (Yout[i]-cCentreY)**2)
        if maxDist < dis:
            maxDist=dis
             
    theta = (math.pi*2)/9.0
    sectors = np.zeros(shape=(9,3,2), dtype=int)
    th=0
    s=0
    xcircle = (maxDist) * math.cos(th) + cCentreX
    ycircle = (maxDist) * math.sin(th) + cCentreY

    clr = (255, 255, 255)
    for i in range (1, 360, 40):
        sectors[s,0,0]=xcircle
        sectors[s,0,1]=ycircle
        th = th + theta
        xcircle = (maxDist+100) * math.cos(th) + cCentreX
        ycircle = (maxDist+100) * math.sin(th) + cCentreY
        sectors[s,1,0]=xcircle
        sectors[s,1,1]=ycircle
        sectors[s,2,0]=cCentreX
        sectors[s,2,1]=cCentreY
        s = s+1
        # shape = [(xcircle-1, ycircle-1), (xcircle+1, ycircle+1)]
        # img1.ellipse(shape, fill=clr, outline=clr, width=0)
        
        # shape = [(xcircle, ycircle), (cCentreX, cCentreY)]
        # img1.line(shape, fill=clr, width=1)
    
    outerpts = np.zeros(shape=(len(Xout),2), dtype=int)
    innerpts = np.zeros(shape=(len(Xin),2), dtype=int)
    for i in range(0, len(Xout)):
        outerpts[i,0]=Xout[i]
        outerpts[i,1]=Yout[i]
    for i in range(0, len(Xin)):
        innerpts[i,0]=Xin[i]
        innerpts[i,1]=Yin[i]
        
    innerRibbonPath = mplPath.Path(innerpts)
    outerRibbonPath = mplPath.Path(outerpts)
   
    # ptssect = np.zeros(shape=(3,2), dtype=int)
    # ptssect[0,0] = sectors[0,0,0]
    # ptssect[0,1] = sectors[0,0,1]
    # ptssect[1,0] = sectors[0,1,0]
    # ptssect[1,1] = sectors[0,1,1]
    # ptssect[2,0] = sectors[0,2,0]
    # ptssect[2,1] = sectors[0,2,1]  
    #sectorPath = mplPath.Path(ptssect) 
    
    #build a histogram of 9X9=81 bins
    lbp = np.zeros(shape=(9*58), dtype=float)
    ulbp = GetULBP58()
    clr=(255, 0,0)
    #calculate HOG for the bounding box!
    binary= np.zeros(shape=(8))
    for x in range(MyImg.width):
        for y in range(MyImg.height):
            p=(x, y)
            contain=outerRibbonPath.contains_point(p)
            if contain == True:
                contain=innerRibbonPath.contains_point(p)
                if contain == False:
                    for s in range(0,9):
                        ptssect = np.zeros(shape=(3,2), dtype=int)
                        for i in range(0,3):
                            ptssect[i,0] = sectors[s,i,0]
                            ptssect[i,1] = sectors[s,i,1]
                            
                        sectorPath = mplPath.Path(ptssect)  
                        contain=sectorPath.contains_point(p)
                        if contain == True:
                            # clr=((s)*28, (s+1)*20, int(255/(s+1)))
                            # img1.point((x, y), fill=clr)
                            
                            #calculate LBP
                            pintensity= MyImg.getpixel((int(x), int(y)))[0]
                            
                            #we calculate LBP for pixels not leying on the four edges
                            if pintensity > MyImg.getpixel((int(x+1), int(y-1)))[0]:
                                binary[0]=0
                            else:
                                binary[0]=1
                                
                            if pintensity > MyImg.getpixel((int(x), int(y-1)))[0]:
                                binary[1]=0
                            else:
                                binary[1]=1
                            if pintensity > MyImg.getpixel((int(x-1), int(y-1)))[0]:
                                binary[2]=0
                            else:
                                binary[2]=1
                                
                            if pintensity > MyImg.getpixel((int(x-1), int(y)))[0]:
                                binary[3]=0
                            else:
                                binary[3]=1
                                
                            if pintensity > MyImg.getpixel((int(x-1), int(y+1)))[0]:
                                binary[4]=0
                            else:
                                binary[4]=1
                                
                            if pintensity > MyImg.getpixel((int(x), int(y+1)))[0]:
                                binary[5]=0
                            else:
                                binary[5]=1
                                
                            if pintensity > MyImg.getpixel((int(x+1), int(y+1)))[0]:
                                binary[6]=0
                            else:
                                binary[6]=1
                                
                            if pintensity > MyImg.getpixel((int(x+1), int(y)))[0]:
                                binary[7]=0
                            else:
                                binary[7]=1
                    
                            #convert the 8 bit code to decimal number
                            decimal=0
                            for k  in range(len(binary)):
                                decimal = int(decimal + binary[k] * math.pow(2, k))

                            for i in range(len(ulbp)):
                                if decimal == ulbp[i]:
                                    lbp[s*58+i] = lbp[s*58+i] + pintensity
                                    break
                            #we found the sector, break the s loop!
                            break
                           
                 
    #now we have the histogram, we need to normalize it
    #we divide each bin by the total of all bins
    # maxbin=0
    # minbin=0
    '''
    sumbinsSec=[]
    for s in range (0, 9):
        sumbinsSec.append(0)
        for bn in range(58):
            sumbinsSec[s] = sumbinsSec[s]+lbp[s*58+bn]
            
    for s in range (0, 9):
        for bn in range(58):
            lbp[s*58+bn] = lbp[s*58+bn] / sumbinsSec[s]
    '''
    #standeralization instead of normalization
    minbins=min(lbp)
    stdbins=np.std(lbp)
    
    for s in range (0, 9):
        for bn in range(58):
            lbp[s*58+bn] = (lbp[s*58+bn] - minbins)/ (stdbins * stdbins)
        
            
        # if maxbin < hog[i] :
        #     maxbin=hog[i]
    
    #and now range normalize it to get all values between 0, 1
    # for i in range (0, len(hog)):
    #     hog[i] = (hog[i] -minbin) / (maxbin - minbin)
        
    return lbp
     
for margin in range(18, 19):
    print('###############\t'+str(margin))
    hogfile = open("D:\\MyData\\DataSet\\ThyroidDS\Thyroid Cancer Signs All Cases\\LBP\\LBP_MarginSectors\\HistogramAddingPixelIntensity\\AllSectors\\9Sectors\\58\\SizeNormEachSector\\lbp_marginStanderized_"+str(margin)+".txt", "w")
    path='E:/MyData/DataSet/ThyroidDS/Thyroid Cancer Signs All Cases/All Original'
    files = os.listdir(path)
    count=0
    for fl_act in files:   
      MyImg = Image.open(path+"/"+fl_act).convert('RGB') 
      
      split=fl_act.split('\\')
      fl=split[len(split)-1]
      fl=fl.split('.')[0]
       
      textfile = open("E:\MyData\DataSet\ThyroidDS\Thyroid Cancer Signs All Cases\ROI_Coordinates434Images.txt", 'r')
      irregfile = open("E:\MyData\DataSet\ThyroidDS\Thyroid Cancer Signs All Cases\Irregularity434ImagesSortedPercentages.txt", 'r')
      fl=os.path.splitext(fl)[0]
      irregOrg=0
          
      while True:
           line = textfile.readline()
           irregline= irregfile.readline()
           
           i=line.find(fl, 0, len(fl))
           if i != -1:
              irregOrg=irregline.split('\t')[1]
              i=line.find('[')
              j=line.find(']')
              #print(line[i + 1: j-1])
              points = (line[i + 1: j -1]).split(';')
              X=np.array([[]], dtype=np.float64)
              Y=np.array([[]], dtype=np.float64)
              
              sumX=0
              sumY=0
              for act in points:
                st=act.split(',')
                X=np.append(X,int(st[0]))                      
                Y=np.append(Y,int(st[1]))
                sumX = sumX + int(st[0])
                sumY = sumY + int(st[1])
              break
       #calculate centroid
      Xcentroid = sumX / len(points)
      Ycentroid = sumY / len(points)

      hog = DrawRibbonSectors_58_LBP(X, Y, margin, MyImg)
      hogfile.write(fl + "\t" + str(int(irregOrg)) + "\t")
      hogfile.write(str(round(hog[0], 5)))
      for i in range(1, len(hog)):
          hogfile.write(',' + str(round(hog[i], 5)))
          
      #display(MyImg)
      pat="D:\\MyData\\DataSet\\ThyroidDS\\Thyroid Cancer Signs All Cases\\CalculateIrregularityGausian\\GausianImages\\Sigma2.0\\"+fl+".png"
      #MyImg.save(pat)
      count=count+1
      print(count)
      hogfile.write('\n')
    hogfile.close()
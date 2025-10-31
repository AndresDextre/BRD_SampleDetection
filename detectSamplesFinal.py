import cv2
import numpy as np
import imutils
#from pyimagesearch.shapedetectory import ShapeDetector
import pandas as pd
import math
import copy
from sklearn.cluster import KMeans

def detectSamples(filename, numSamples):
  #Import image

  image = cv2.imread(filename)
  height, width, color = image.shape
  
  # Convert BGR to HSV
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  
  # define range of blue color in HSV
  #negative
  lower_neg = np.array([140,70,0])
  upper_neg = np.array([179,255,255])
  secondlower_neg = np.array([0,70,0])
  secondupper_neg = np.array([15,255,255])
  
  #positive
  lower_pos = np.array([0,0,0])
  upper_pos = np.array([0,255,255])
  
  maskneg = cv2.inRange(hsv, lower_neg, upper_neg)
  maskneg2 = cv2.inRange(hsv,secondlower_neg,secondupper_neg)
  #maskpos = cv2.inRange(hsv, lower_pos, upper_pos)
  negativeMask = maskneg | maskneg2
  negOutput = cv2.bitwise_and(image, image, mask=negativeMask)
  #posOutput = cv2.bitwise_and(image, image, mask=maskpos)
  
  combinedOutput = negOutput #+ posOutput
  # cv2.imshow("combined output", combinedOutput)
  # cv2.waitKey(0)
  
  #Resize image for better shape approximation
  resized = imutils.resize(combinedOutput, width=500) #original width 300
  ratio = image.shape[0] / float(resized.shape[0])
  # cv2.imshow("resized", resized)
  # cv2.waitKey(0)
  
  # convert the resized image to grayscale, blur it slightly,
  # and threshold it
  gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (3, 3), 0)
  thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1]
  
  # find contours in the thresholded image and initialize the
  # shape detector
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  cnts = imutils.grab_contours(cnts)
  validContours = [c for c in cnts if (cv2.contourArea(c) > 5 and cv2.contourArea(c) < 5000)]
  #k2 = np.ones((5,1),np.uint8)
  #for c in range(len(validContours)):
  #  curve = 0.01 * cv2.arcLength(validContours[c],True)
  #  validContours[c] = cv2.approxPolyDP(validContours[c],curve,True)
  #'''
  cv2.imshow('thresh',thresh)
  iterate = 1
  while len(validContours) < numSamples:
    areas = [cv2.contourArea(c) for c in validContours]
    index = areas.index(max(areas))
    mergedContour = validContours[index]
    array = np.vstack(mergedContour)
    all_points = array.reshape(array.shape[0], array.shape[1])
    nclust = 2
    if iterate == 1:
      nclust = 2
    kmeans = KMeans(n_clusters=nclust, random_state=0, n_init = 100).fit(all_points)
    new_contours = [all_points[kmeans.labels_==i] for i in range(nclust)]
    
    del validContours[index]
    for q in new_contours:
      q = q.astype(np.int32)
      shapeq = q.shape
      q = q.reshape([shapeq[0],1,shapeq[1]])
      validContours.append(q)

    
    #points = [Point(x[0],x[1]) for x in approxContour]
    #vor = Voronoi(approxContour)
    #points = vor.points
    #points = [Point(x[0],x[1]) for x in points]
    #P1, P2 = closest(points,len(points))
    #roi = thresh[P1[1]-3:P2[1]+3,P1[0]-3:P2[0]+3]
    
    iterate+=1

  #cv2.imshow('thresh',thresh)
  results = []
  #loop over each contour
  count = 0
  for c in validContours:
      M = cv2.moments(c)
      if M["m00"] != 0:
          cX = int((M["m10"] / M["m00"]) * ratio)
          cY = int((M["m01"] / M["m00"]) * ratio)
      
  
      #cv2.imshow('thresh', thresh)
  
  
  
      # multiply the contour (x, y)-coordinates by the resize ratio,
      # then draw the contours and the name of the shape on the image
      c = c.astype("float")
      c *= ratio
      c = c.astype("int")
      
      #print(f'Sample {count} is at x: {cX} and y: {cY}')
      # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
  
      # if (count < 5):
      #Find x,y range of sample
      minX = 999999
      maxX = 0
      minY = 999999
      maxY = 0
      for loc in c:
          if (loc[0][0] < minX):
              minX = loc[0][0]
          if (loc[0][0] > maxX):
              maxX = loc[0][0]
          if (loc[0][1] < minY):
              minY = loc[0][1]
          if (loc[0][1] > maxY):
              maxY = loc[0][1]
  
      redCount = 0
      # print(f'X: {minX}-{maxX}  Y: {minY}-{maxY}')
      # print(negOutput.shape)

      for x in range(minY, maxY):
          for y in range(minX, maxX):
              if (negOutput[x, y, 0] != 0 or negOutput[x, y, 1] != 0 or negOutput[x, y, 1] != 0):
                  redCount += 1
      yellowCount = 0
      for x in range(minY, maxY):
          for y in range(minX, maxX):
              if (posOutput[x, y, 0] != 0 or posOutput[x, y, 1] != 0 or posOutput[x, y, 1] != 0):
                  yellowCount += 1
      
      positivity = yellowCount / (redCount + yellowCount) * 100

      results.append(c)

      cv2.putText(image, f'{count}', (cX+25, cY+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
      cv2.drawContours(image, [c], -1, (255, 0, 0), 2)

      count += 1
  # show the output image
  #cv2.imshow('images', np.hstack([combinedOutput, image]))
  image = imutils.resize(image, width=500)
  cv2.imshow("Image", image)
  
    
  while(1):
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
      cv2.destroyAllWindows()  
      break
  cv2.imwrite('output'+filename,image)
  #results = pd.DataFrame(results,columns=[filename])
  # cv2.waitKey(0)
  return validContours

def usePrevious(name,cont):
  image = cv2.imread(name)
  height, width, color = image.shape
  
  # Convert BGR to HSV
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  
  # define range of blue color in HSV
  #negative
  lower_neg = np.array([150,0,0])
  upper_neg = np.array([179,255,255])
  secondlower_neg = np.array([0,0,0])
  secondupper_neg = np.array([10,255,255])
  
  #positive
  lower_pos = np.array([10,0,0])
  upper_pos = np.array([40,255,255])
  
  maskneg = cv2.inRange(hsv, lower_neg, upper_neg)
  maskneg2 = cv2.inRange(hsv,secondlower_neg,secondupper_neg)
  maskpos = cv2.inRange(hsv, lower_pos, upper_pos)
  negativeMask = maskneg | maskneg2
  negOutput = cv2.bitwise_and(image, image, mask=negativeMask)
  posOutput = cv2.bitwise_and(image, image, mask=maskpos)
  combinedOutput = negOutput + posOutput
  resized = imutils.resize(combinedOutput, width=500) #original width 300
  ratio = image.shape[0] / float(resized.shape[0])
  count = 0
  results = []
  for c in cont:#validContours:
      M = cv2.moments(c)
      if M["m00"] != 0:
          cX = int((M["m10"] / M["m00"]) * ratio)
          cY = int((M["m01"] / M["m00"]) * ratio)

      #cv2.imshow('thresh', thresh)

      # multiply the contour (x, y)-coordinates by the resize ratio,
      # then draw the contours and the name of the shape on the image
      c = c.astype("float")
      c *= ratio
      c = c.astype("int")
      
      #print(f'Sample {count} is at x: {cX} and y: {cY}')
      # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
  
      # if (count < 5):
      #Find x,y range of sample
      minX = 999999
      maxX = 0
      minY = 999999
      maxY = 0
      for loc in c:
          if (loc[0][0] < minX):
              minX = loc[0][0]
          if (loc[0][0] > maxX):
              maxX = loc[0][0]
          if (loc[0][1] < minY):
              minY = loc[0][1]
          if (loc[0][1] > maxY):
              maxY = loc[0][1]
  
      redCount = 0
      for x in range(minY, maxY):
          for y in range(minX, maxX):
              if (negOutput[x, y, 0] != 0 or negOutput[x, y, 1] != 0 or negOutput[x, y, 1] != 0):
                  redCount += 1
      yellowCount = 0
      for x in range(minY, maxY):
          for y in range(minX, maxX):
              if (posOutput[x, y, 0] != 0 or posOutput[x, y, 1] != 0 or posOutput[x, y, 1] != 0):
                  yellowCount += 1
      positivity = yellowCount / (redCount + yellowCount) * 100
      results.append(positivity)
      #cv2.putText(image, f'{count}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
      #cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
      #count += 1
      #cv2.putText(image, 'sample', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 10)
   
  #cv2.imshow("Image", image)
  
  '''  
  while(1):
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
      cv2.destroyAllWindows()  
      break
  # show the output image
  #cv2.imshow('images', np.hstack([combinedOutput, image]))
  '''
  return results

results = detectSamples('20211026_00min_HS_NarrowRangeLOD_Low.png',96)
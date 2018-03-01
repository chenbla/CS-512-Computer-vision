# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 01:12:32 2017

@author: Prateek
"""


import cv2
import sys
import numpy as np
import math







def nothing(x):
    pass



def compute_harris(frame, sigma, k) :
    

    rows = frame.shape[0]
    cols = frame.shape[1]

    cov = np.zeros((rows,cols * 3), dtype = np.float32)
    dst = np.zeros((rows,cols), dtype = np.float32)
    #computing x and y derivative of image
    dx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    dy = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)

    ksize = max(5,  5 * sigma)
    if (ksize %2 == 0 ) :
        ksize = ksize + 1



    #computing products of derivatives at every pixel
    Ixx = cv2.GaussianBlur(dx*dx,(ksize,ksize),sigma)
    Ixy = cv2.GaussianBlur(dx*dy,(ksize,ksize),sigma)
    Iyy = cv2.GaussianBlur(dy*dy,(ksize,ksize),sigma)

    for i in range(0, rows,1) :
            for j in range(0, cols,1) :
                a = cov[i, j*3] =   Ixx[i,j]
                b = cov[i, j*3+1] = Ixy[i,j]
                c = cov[i, j*3+2] = Iyy[i,j]
                dst[i,j] = a*c - b*b - k*(a+c)*(a+c)

    return dst

def cornerSubPix(src, centroids, block_size):
  

    output = np.zeros((centroids.shape[0]-1,centroids.shape[1],1), dtype=np.float32 )

    win_w = block_size * 2 + 1
    win_h = block_size * 2 + 1


    # pixel position in the rectangle
    x = np.arange(-block_size, block_size +1, dtype=np.int)
    y = np.arange(-block_size, block_size +1, dtype=np.int)

    # do optimization loop for all centroids
    for i in range(1,len(centroids)) :
        im = cv2.getRectSubPix(src,(win_w, win_h), (centroids[i][0],centroids[i][1]))

        # 1st derivative of image
        dx = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=5)
        dy = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=5)

        # dx2,dy2, dxy
        Ixx = dx**2
        Ixy = dy*dx
        Iyy = dy**2

        #sum of the Ixx, Iyy, Ixy
        Sxx = np.sum(Ixx,None)
        Sxy = np.sum(Ixy,None)
        Syy= np.sum(Iyy,None)

        #sum I(xi)I(xi)t*xi
        bb1 = Ixx * x + Ixy * y
        bb2 = Ixy * x + Iyy * y

        #c-1 * sum I(xi)I(xi)t*xi
        det = Sxx*Syy - Sxy**2
        scale =  1/det

        output[i-1][0] = centroids[i][0] + Syy*scale* np.sum(bb1,None) - Syy*scale * np.sum(bb2,None)
        output[i-1][1] = centroids[i][1] - Sxy*scale* np.sum(bb1,None) + Sxx*scale * np.sum(bb2,None)

    return  output

def corner_harris_and_localization(image1,sigma,k,block_size,threshold_value ) :
        # convert image1 to gray
        gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

       
        dst = compute_harris(gray, sigma, k)

        print('threshold value', threshold_value)
        print('dst.max', dst.max())
        print('threds total', 0.001 * threshold_value *dst.max())
        ret, dst1 = cv2.threshold(dst,0.001 * threshold_value *dst.max(),255,0)

        # find centroids of the points around the area
        dst2 = np.uint8(dst1)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst2)

        #localization
        corners = cornerSubPix(gray, np.float32(centroids), block_size)
        return corners

def corner_featureVector(corners, image1, block_size):
   

    # convert image1 to gray
    gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    win_w = block_size * 2 + 1
    win_h = block_size * 2 + 1

    
    degree_hist = np.zeros((len(corners),9,1), dtype=np.uint8)

    for i in range(0,len(corners)) :
        im = cv2.getRectSubPix(gray,(win_w, win_h), (corners[i][0],corners[i][1]))

        # 1st derivative of image
        dx = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=5)
        dy = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=5)

        angle = np.arctan2(dy,dx)

       

        for j in range(0, 9) :
            degree_hist[i][j] = ((( math.pi/4 * (j-4))<= angle) & (angle < (math.pi/4 *(j-3)))).sum()

    return degree_hist

def Corner_featurePoints(image1, image2, c1,c2,hist1,hist2) :
    

    # draw rectangle of the corner
    c1 = np.int0(c1)
    c2 = np.int0(c2)

    # draw an empty rectangle around the corner
    for i in range(0,c1.shape[0]) :
          cv2.rectangle(image1, (c1[i,0] - 18, c1[i,1]-18), (c1[i,0]+18, c1[i,1]+18), [0,0,0])
    for j in range(0,c2.shape[0]) :
          cv2.rectangle(image2, (c2[j,0] - 18, c2[j,1]-18), (c2[j,0]+18, c2[j,1]+18), [0,0,0])

    cno = 1
    min_distance = 0
    min_index = 0

    for i in range(0, len(c1)):
        for j in range(0, len(c2)) :
                # compare histogram for each corner,find the smallest distance between corner in image 1 and image 2.
                distance = np.sum( (hist1[i] - hist2[j] ) * (hist1[i] - hist2[j] ),None)
                if (j ==0 ) :
                    min_distance = distance
                    min_index = j
                else  :
                    if distance < min_distance :
                        min_distance = distance
                        min_index = j

        cv2.putText(image1,str(cno), (c1[i,0],c1[i,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(image2,str(cno), (c2[min_index,0],c2[min_index,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2,cv2.LINE_AA)
        cno = cno +1

    cv2.imshow('image1',image1)
    cv2.imshow('image2',image2)


def process(og1, og2) :
    
    print("Press 'h' for implementing Harris Corner algorithm")
        

    while True:
        key = cv2.waitKey(0)
        if key == ord('h') :
            
            cv2.namedWindow('corner')

            #Track Bar
            trace_bar = np.zeros((120,320,1), dtype=np.float32 )
            cv2.createTrackbar('Variance','corner',5,5,nothing)
            cv2.createTrackbar('Neighborhood Size','corner',2,4,nothing)
            cv2.createTrackbar('Trace','corner',4,15,nothing)
            cv2.createTrackbar('Threshold','corner',10,100,nothing)


            cv2.imshow('corner',trace_bar)

            while (True) :
                

                sigma = cv2.getTrackbarPos('Variance','corner')
                block_size = cv2.getTrackbarPos('Neighborhood Size','corner')
                k = cv2.getTrackbarPos('Trace','corner')
                threshold_value = cv2.getTrackbarPos('Threshold','corner')
                image1=og1.copy()
                image2=og2.copy()

                #threshold_value = 1

                sigma = 0.01 * sigma

                k = 0.01 * k

                c1 = corner_harris_and_localization(image1,sigma,k,block_size,threshold_value)
                hist1 = corner_featureVector(c1, image1, block_size)

                c2 = corner_harris_and_localization(image2,sigma,k,block_size,threshold_value)
                hist2 = corner_featureVector(c2, image2, block_size)

                Corner_featurePoints(image1,image2,c1,c2,hist1,hist2)
                

                if cv2.waitKey(5) == 27 :
                    cv2.destroyAllWindows()
                    break

def main():
    

        global image1
        global image2

        img = cv2.imread('test2.png')
        img1 = cv2.imread('test2.png')

        cv2.imshow('image1',img)
        cv2.imshow('image2',img1)

        og1=img.copy()
        og2=img1.copy()

        process(og1,og2)

if __name__ == '__main__':
    main()


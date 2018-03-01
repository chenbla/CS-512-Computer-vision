import cv2
import os
from skimage import io, color
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


#Welcome to Assignment 2

userinput=int(input('Enter 1-->Img capture 2-->predefined image :'))
if(userinput==1):
    try: 
        os.remove("frame1.jpg")
    except:
        pass
    vidcap = cv2.VideoCapture(0)
    success,image = vidcap.read()
    count = 0
    if(count==0):
        success = True
        count+=1   
    while success:
        success,image = vidcap.read()
        cv2.imwrite("frame%d.jpg" % count, image)
#        imgloc = r"D:\homework and assignments\computer vision\opencv python\frame1.jpg"
        vidcap.release()
        if(count>0):
            success=False 
    img=cv2.imread('frame1.jpg')
#    img2=cv2.imread('frame1.jpg')
    imggray=cv2.imread('frame1.jpg',0)
            
if(userinput==2):
    img = cv2.imread('bmw.jpg')
#    img2=cv2.imread('bmw.jpg')
    imggray=cv2.imread('bmw.jpg',0)
else:
    pass
ip=1
print('PRESS h for help about the functions related to the keys')
while(ip==1):
    x=str(input('Enter your choice : '))
    if(x=='i'):
        cv2.imwrite('normal.jpg',img)   
        img1=cv2.imread('normal.jpg')
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ip=1
    if(x=='w'):
         cv2.imwrite('out.jpg',img)
         ip=1
    if(x=='g'):
        cv2.imwrite('gray.jpg',img)
        img1=cv2.imread('gray.jpg',0)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ip=1
    if(x=='G'):
        cv2.imwrite('normal.jpg',img)   
        img1=cv2.imread('normal.jpg')
        gray = img[:,:,1]
        
       
        
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ip=1
            
    if(x=='c'):
        cv2.imwrite('normal.jpg',img)   
        img1=cv2.imread('normal.jpg')
        x1=str(input('Enter sub choice 1-->red 2-->green 3--> blue :'))
        if(x1=='1'):
            print('Red channel')
            img1[:,:,0] = 0
            img1[:,:,1] = 0
            out=img1
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image',out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ip=1
        if(x1=='2'): 
            print('Green channel')
            img1[:,:,0] = 0
            img1[:,:,2] = 0
            out=img1
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image',out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ip=1
        if(x1=='3'):
            print('Blue channel')
            img1[:,:,1] = 0
            img1[:,:,2] = 0
            out=img1
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image',out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ip=1
            
    if(x=='s'):
        def nothing(x):
            pass
        cv2.imwrite('normal.jpg',img)   
        img1=cv2.imread('normal.jpg')
        blur=cv2.imread('normal.jpg')
        cv2.namedWindow('test')
        n=int(input('Enter the filter size----->'))
        while(1):
            blur=img1
            print('Press enter to select the value you have slided')
            cv2.createTrackbar('smooth','test',0,n,nothing)
            print('press esc to close the window')
            ch=cv2.waitKey(0) & 0xff
            x=cv2.getTrackbarPos('smooth','test')
            for i in range(1,x):
                blur=cv2.blur(blur,(n,n))
                cv2.imshow('test',blur)
            if ch==27:
                break
    cv2.destroyAllWindows()
    ip=1    
        
    if(x=='S'):
        
        def convolve2d(image, kernel):
            kernel = np.flipud(np.fliplr(kernel))    
            output = np.zeros_like(image)            
        # Add zero padding to the input image
            image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
            image_padded[1:-1, 1:-1] = image
            for x in range(image.shape[1]): 
                # Loop over every pixel of the image
                for y in range(image.shape[0]):
                    output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
            return output
        
        cv2.imwrite('gray.jpg',img)
        img1=cv2.imread('gray.jpg',0)
        kernel = (np.array([[1,1,1],[1,1,1],[1,1,1]])/9)
        blur= convolve2d(img1,kernel)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ip=1
    if(x=='d'):
        imgnew=cv2.pyrDown(img,2)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',imgnew)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ip=1
    if(x=='D'):
        blur = cv2.blur(img,(3,3))
        imgnew=cv2.pyrDown(blur,2)
       
        cv2.imshow('image',imgnew)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ip=1
    if(x=='x'):
        cv2.imwrite('gray.jpg',img)
        img1=cv2.imread('gray.jpg',0)
        der=[[-1,0,1],[-2,0,2],[-1,0,1]]
        xder=np.array(der)
        normImg=np.zeros((1920,1080))
        newImgx=ndimage.convolve(img1,xder,mode='constant')
        normaImgx=cv2.normalize(newImgx,normImg,0,255,cv2.NORM_MINMAX)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',normaImgx)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ip=1
        
    if(x=='y'):
        cv2.imwrite('gray.jpg',img)
        img1=cv2.imread('gray.jpg',0)
        der=[[-1,-2,-1],[0,0,0],[1,2,1]]
        xder=np.array(der)
        normImg=np.zeros((1920,1080))
        newImgy=ndimage.convolve(img1,xder,mode='constant')
        normaImgy=cv2.normalize(newImgy,normImg,0,255,cv2.NORM_MINMAX)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',normaImgy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ip=1
    
    if(x=='p'):
        cv2.imwrite('gray.jpg',img)
        img1=cv2.imread('gray.jpg',0)
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
        plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([]) 
        plt.show()
        ip=1
        
    if(x=='r'):
        theta=float(input('Enter an angle --->' ))
        cv2.imwrite('gray.jpg',img)
        img1=cv2.imread('gray.jpg',0)
        num_rows, num_cols = img1.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), theta, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
        cv2.imshow('Rotation', img_rotation)
        cv2.waitKey()
        ip=1
        
    if(x=='m'):
        def gramag(x,y,img):
            sobel1=cv2.Sobel(img,cv2.CV_64F,x,y,ksize=5)
            abs_val=np.absolute(sobel1)
            s8U=np.uint8(abs_val)
            return s8U
        cv2.imwrite('normal.jpg',img)   
        img1=cv2.imread('normal.jpg')
        op=gramag(1,1,img1)
        cv2.imshow('Image',op)
        cv2.waitKey()
        
       
        
    if(x=='h'):
        print('\n Press i to view the input image \n Press w to save the image \n Press g to view the grayscale image \n Press G to view the custom grayscale image \n Press i to view the input image \n Press c to view the image in various color channels\n Press s to smooth the image with trackbar functionality \n Press S to view custom smmothing of image \n Press d to downsample the image without smoothing \n Press D to smooth the image and then downsample \n Press x to perform convolution with x derivative filter with normalization\n Press y to perform convolution with y derivative filter with normalization \n Press p to convert image to grayscale and plot the gradient vectors \n Press r to rotate the image with an angle theta \n')
        ip=0

    
    
    
   

     
       
    
   
    
     
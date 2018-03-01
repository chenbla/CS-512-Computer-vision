import cv2
import numpy as np
from matplotlib import pyplot as plt
import Fundamental as Mat
global pnts1
global pnts2
global pnts3
global pnts4
pnts3=[]
pnts4=[]
pnts5=[]
pnts6=[]







def drawlines(img1,img2,lines,pnts1,pnts2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pnts1,pnts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1),10,color,-1)
        cv2.circle(img2,tuple(pt2),10,color,-1)

    return img1,img2

def my_mouse_callbackL(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img1,(x, y), 5, (0, 0, 255), -1)
     pnts3.append([x,y])
     pnts5.append([x,y])
     
     print(x, y)

  
  
  
def my_mouse_callbackR(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img2,(x, y), 5, (0, 0, 255), -1)
     pnts4.append([x,y])
     pnts6.append([x,y])
     print(x, y)
     
     
     
     
def my_mouse_callbackL1(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img1,(x, y), 5, (0, 0, 255), -1)
     index1=pnts5.index([x,y])
     img7,img8=draw_single_Line(imgx,imgy,lines1,pnts5[index1],pnts5[index1])
     plt.subplot(121),plt.imshow(img7)
     plt.subplot(122),plt.imshow(img8)
     plt.show()
     
  
  
  
def my_mouse_callbackR1(event,x,y,flags,param):
 if event == cv2.EVENT_LBUTTONDOWN:
     cv2.circle(img2,(x, y), 5, (0, 0, 255), -1)
     index2=pnts5.index([x,y])
     img9,img10=draw_single_Line(imgy,imgx,lines1,pnts5[index2],pnts5[index2])
     plt.subplot(121),plt.imshow(img9)
     plt.subplot(122),plt.imshow(img10)
     plt.show()
     
def draw_single_Line(img1,img2,lines,pnts1,pnts2):
    
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r in lines:
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pnts1),10,color,-1)
        cv2.circle(img2,tuple(pnts2),10,color,-1)

    return img1,img2








img1 = cv2.imread('./data/l.tif',0)  
img2 = cv2.imread('./data/r.tif',0) 
imgx = cv2.imread('./data/l.tif',0)  
imgy = cv2.imread('./data/r.tif',0) 
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)


orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
print(des1, des2)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# Match descriptors.
matches = bf.match(des1,des2)

good = []
pnts1 = []
pnts2 = []

D_MATCH_THRES = 65.0
for m in matches:
    if m.distance < D_MATCH_THRES:
        good.append(m)
        pnts2.append(kp2[m.trainIdx].pt)
        pnts1.append(kp1[m.queryIdx].pt)

pnts1 = np.float32(pnts1)
pnts2 = np.float32(pnts2)

# compute F
F = Mat.fundamental_matrix(pnts1,pnts2) 
print(F)
print('SELECT ATLEAST 8 POINTS IN THE IMAGE')
# We select only inlier points

pnts1 = np.array([[0,0]])
pnts2 = np.array([[0,0]])


cv2.setMouseCallback('image1',my_mouse_callbackL)
cv2.setMouseCallback('image2',my_mouse_callbackR)
while True:
 cv2.imshow("image1",img1)
 cv2.imshow("image2",img2)
 if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pnts3=np.float32(np.array(pnts3))
pnts4=np.float32(np.array(pnts4))
funMat=Mat.fundamental_matrix(pnts3,pnts4)
print(funMat)


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pnts4.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pnts3,pnts4)



# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pnts3.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pnts4,pnts3)


plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

print("The coordinates of image 1 are",pnts5)
print("The coordinates of image 2 are",pnts6)

cv2.namedWindow('imagex',cv2.WINDOW_NORMAL)
cv2.namedWindow('imagey',cv2.WINDOW_NORMAL)

cv2.setMouseCallback('imagex',my_mouse_callbackL1)
cv2.setMouseCallback('imagey',my_mouse_callbackR1)
while True:
 cv2.imshow("imagex",imgx)
 cv2.imshow("imagey",imgy)
 if cv2.waitKey(1) & 0xFF == ord('q'):
        break





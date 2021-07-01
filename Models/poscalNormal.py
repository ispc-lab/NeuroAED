import cv2
import numpy as np
from skimage import measure

font=cv2.FONT_HERSHEY_COMPLEX

def poscalNormal(im):
    im_labels = measure.label(im,connectivity=1,neighbors= 8)  #tag each connected component from 0
    num = im_labels.max()     # number of connected components
    if num==0:
        im_s = np.zeros((1,5))   # if ab_fg_img is NONE
    else:
        im_s = np.zeros((num,5))
        for i in range(num):
            temp = np.copy(im_labels)
            temp[temp != (i+1)]=0
            index = np.where(temp ==(i+1))
            im_s[i,0]= max(index[0]) #person's foot y_val
            im_s[i,1]= min(index[0]) #person's head y_val
            im_s[i,2]= max(index[1]) #person's right side x_val
            im_s[i,3]= min(index[1]) #person's left side x_val
            im_s[i,4]= len(index[0]) #area of the person
    return im_labels,num,im_s,im


def labeling(pos,abnormal_fg_img):
    gate=0.4
    labels=[]
    labeledPos=np.zeros((0,5))
    for thePos in pos:
        if abnormal_fg_img[int(thePos[1]):int(thePos[0]),int(thePos[3]):int(thePos[2])].any():
            #if the fulfillment from abnormal image in this tagged area is above a certain gate, tag 1
            if abnormal_fg_img[int(thePos[1]):int(thePos[0]),int(thePos[3]):int(thePos[2])].mean()>gate:
                labeledPos=np.concatenate((labeledPos,thePos.reshape(1,-1)))
                labels.append(1)
        else:
            labeledPos = np.concatenate((labeledPos, thePos.reshape(1, -1)))
            labels.append(0)

    return  labeledPos,np.array(labels)

def save_abnormal_Images(save_path):
    img = cv2.imread(save_path)
    cv2.rectangle(img, (0,10),  (40, 30), (255,0,0), thickness=-1)
    cv2.putText(img, 'Abnormal', (10,10), font, 0.6, (255, 255, 0), 2)
    cv2.imwrite(save_path)


def save_detection_Images(img1,save_path):
    img2 = cv2.imread(save_path)
    _,_,im_s,_=poscalNormal(img1)
    for i,item in enumerate(im_s):
        cv2.rectangle(img2,(int(item[3]),int(item[1])),(int(item[2]),int(item[0])),(0, 0, 255),2)
        cv2.putText(img2, str(i), (int(item[3]),int(item[1])-5), font, 0.4, (255, 255, 0), 1)
    cv2.imwrite(save_path, img2)

import cv2
from poscalNormal import poscalNormal
from poscalNormal import labeling

def getPosition(forgpic):
        # this_Spliter = Spliter()
        # weight = self.weigh
        ab_img = cv2.imread(forgpic,0)
        labels,num,im_s,mopho_img = poscalNormal(ab_img)
        # splitPos = this_Spliter.split(im_s,mopho_img,weight)
        # realPos, label = labeling(im_s,ab_img)

        return labels,num

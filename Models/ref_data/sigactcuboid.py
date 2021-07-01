import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# from GlobalFeatureExtraction import readFrames
from utils import dictFeatureName
import cv2
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from pixel_operation import getPosition
from poscalNormal import save_detection_Images
from poscalNormal import save_abnormal_Images


def readFrames(directory):
    videos = []
    videos = [name for name in os.listdir(directory)]
    videos.sort()
    i=0
    for video in videos:
        names=os.listdir(os.path.join(directory,video))
        names.sort()
        frameNames = [os.path.join(directory,video,name) for name in names ]
        frameTime =  [int(os.path.splitext(name)[0]) for name in names]
       
        if i==0:
            data_frameNames = np.array(frameNames)
            data_frameTimes = np.array(frameTime)
        else:
            data_frameNames = np.row_stack((data_frameNames.reshape(-1,1),np.array(frameNames).reshape(-1,1)))
            data_frameTimes = np.row_stack((data_frameTimes.reshape(-1,1),np.array(frameTime).reshape(-1,1)))
        i+=1
    return data_frameNames,data_frameTimes


class SigActCuboid(object):
    def __init__(self,dataset_dir):
        self.dataset_name = 'sigactcuboid'
        self.feature_name = os.path.split(dataset_dir)[1]
        self.model_dir = os.path.join(os.path.split(dataset_dir)[0],'unnorm_models')
        self.train_path = os.path.join(dataset_dir,'nor')
        self.test_path = os.path.join(dataset_dir,'abnor')
        self.labels_path = os.path.join(os.path.split(dataset_dir)[0],'labels')
        self.save_path = os.path.join(os.path.split(dataset_dir)[0],'results')
        self.pixels_png =  os.path.join(os.path.split(dataset_dir)[0],'pixels_png')
        
        if self.feature_name=='appearance':
            self.train_video=os.path.join(dataset_dir,'train_video')
            self.test_video=os.path.join(dataset_dir,'test_video')

        else:
            self.train_video=None
            self.test_video=None
        
        
        self.train_data={}
        self.val_data={}
        self.test_data={}
        self.load_data()

        self.test_acc=np.zeros((len(self.test_data['labels']), 1))#1列
        
    
    def read_data(self, feature_file, separator='  '):
        os.chdir(feature_file)
        files = glob.glob("*")
        files.sort()
        i=0
        for f in files:
            if type(f) is str:
                feature =np.loadtxt(f,dtype=np.float32)
                if i==0:
                    data=feature
                else:
                    data=np.row_stack((data,feature))
                i+=1
        return data

    def load_data(self):
        my_imputer = SimpleImputer()
        all_labels = self.read_data(self.labels_path,' ')
        all_times = all_labels[:,0]
        all_labels = all_labels[:,1].T
        all_frame_names, all_frame_times = readFrames(self.pixels_png)

        if self.train_video is None:
            train_data=self.read_data(self.train_path)
            self.train_data['timestamp']=train_data[:,0]
            if self.feature_name == 'SB':
                self.train_data['poses']=None
                self.train_data['vectors']=my_imputer.fit_transform(train_data[:,1:train_data.shape[1]])
            else:
                self.train_data['poses']=train_data[:,1:3]
                self.train_data['vectors']=my_imputer.fit_transform(train_data[:,3:train_data.shape[1]])


            self.train_data['labels']=np.array([0 for i in range(train_data.shape[0])])

            test_data=self.read_data(self.test_path)
            self.test_data['timestamp']=test_data[:,0]
            if self.feature_name == 'SB':
                self.test_data['poses']=None
                self.test_data['vectors']=my_imputer.fit_transform(test_data[:,1:test_data.shape[1]])
            else:
                self.test_data['poses']=test_data[:,1:3]
                self.test_data['vectors']=my_imputer.fit_transform(test_data[:,3:test_data.shape[1]])
            
            test_labels=[]
            test_frames_name=[]
            test_frames_time=[]

            timestamps,ind=np.unique(self.test_data['timestamp'], return_index=True)
            timestamps=timestamps[np.argsort(ind)]
            for timestamp in timestamps:
                idx=np.abs((timestamp-all_times))<50000
                # idx1 = (timestamp - all_times)>0
                # idx = idx0 * idx1
                test_labels.append(all_labels[idx][0])

                test_frames_time.append(all_frame_times[idx][0])
                test_frames_name.append(all_frame_names[idx][0])

            self.test_data['labels'] = np.array(test_labels).reshape(1,-1)
            self.test_data['pixel_labels_times']=np.array( test_frames_time).reshape(1,-1)
            self.test_data['pixel_labels_names']=np.array( test_frames_name).reshape(1,-1)
            self.test_data['pred_labels'] = None
        else:

            feature_train=self.read_data(self.train_path)
            self.train_data['timestamp']=feature_train[:,0]
            self.train_data['poses']=feature_train[:,1:3]
            train_data=readFrames(self.train_video,'nor',self.train_data['timestamp'],self.train_data['poses'])
            self.train_data['vectors']=my_imputer.fit_transform(train_data)
            self.train_data['labels']=np.array([0 for i in range(train_data.shape[0])])

            feature_test=self.read_data(self.test_path)
            self.test_data['timestamp']=feature_test[:,0]
            self.test_data['poses']=feature_test[:,1:3]
            test_data=readFrames(self.test_video,'abnor',self.test_data['timestamp'],self.test_data['poses'])
            self.test_data['vectors']=my_imputer.fit_transform(test_data)

            test_labels=[]
            timestamps,ind=np.unique(self.test_data['timestamp'], return_index=True)
            timestamps=timestamps[np.argsort(ind)]
            for timestamp in timestamps:
                idx=all_times==timestamp
                test_labels.append(all_labels[idx])
            self.test_data['labels'] = np.array(test_labels).reshape(1,-1)
            self.test_data['pred_labels'] = None

        self.diag={}
        self.diag['OneClassSVM']={}
        self.diag['SRC']={}
        self.diag['AutoEncoder']={}

        self.diag['OneClassSVM']['scores']=None
        self.diag['SRC']['scores']=None
        self.diag['AutoEncoder']['scores']=None

        self.diag['OneClassSVM']['auc']=None
        self.diag['SRC']['auc']=None
        self.diag['AutoEncoder']['auc']=None

        self.diag['OneClassSVM']['labels']=None
        self.diag['SRC']['labels']=None
        self.diag['AutoEncoder']['labels']=None

        self.val_data['timestamp']=None
        self.val_data['poses']=None
        self.val_data['vectors']=None
        self.val_data['labels'] = None
        # self.val_data['pred_labels'] = None
        # self.val_data['pred_results'] = None

        
        dataset={'train_data':self.train_data['vectors'],'train_labels': self.train_data['labels'],'train_timestamp':self.train_data['timestamp'],\
                     'val_data':self.val_data['vectors'],'val_labels':self.val_data['labels'],'val_timestamp':self.val_data['timestamp'],\
                     'test_data': self.test_data['vectors'], 'test_labels':self.test_data['labels'],'test_timestamp':self.test_data['timestamp'],}
        self.dataset=dataset
        
    def get_data(self):
        return self.dataset


    def get_pred_results(self, pred_labels):
        self.test_data['pred_labels']=pred_labels
    

    def get_tpr_fpr(self,cub_labels,scores,which_level,name):
        all_tpr=[]
        all_fpr=[]
        thresholds=np.unique(scores)
        thresholds_=[]

        for i in range(0,len(thresholds),200):
            labels=np.zeros((len(scores)))
            threshold=thresholds[-1-i]
            for n in range(len(scores)):
                if scores[n]>threshold:
                    labels[n]=1

            if which_level is 'frame_level':
                tpr,fpr=self.compute_frame_tpr_fpr(labels,flag=True)
            if which_level is 'pixel_level':
                tpr,fpr=self.compute_pixel_tpr_fpr(labels,flag=True)
            all_tpr.append(tpr)
            all_fpr.append(fpr)
            thresholds_.append(threshold)

        labels=np.zeros((len(scores)))
        threshold=thresholds[0]
        for n in range(len(scores)):
            if scores[n]>threshold:
                labels[n]=1

       
        if which_level is 'frame_level':
            tpr,fpr=self.compute_frame_tpr_fpr(labels,flag=False)
        if which_level is 'pixel_level':
            tpr,fpr=self.compute_pixel_tpr_fpr(labels,flag=False)
        all_tpr.append(tpr)
        all_fpr.append(fpr)
        thresholds_.append(threshold)

        all_tpr=np.array(all_tpr)
        all_fpr=np.array(all_fpr)
        thresholds_=np.array(thresholds_)
        idx=np.argsort(all_fpr)

        all_fpr_value=np.zeros((all_fpr.shape[0]))
        all_tpr_value=np.zeros((all_fpr.shape[0]))
        thresholds_value=np.zeros((all_fpr.shape[0]))

        for i in range(all_fpr.shape[0]):
            all_fpr_value[i]=all_fpr[idx[i]]
            all_tpr_value[i]=all_tpr[idx[i]]
            thresholds_value[i]=thresholds_[idx[i]]
        


        if which_level is 'frame_level':
            filename=os.path.join(self.save_path,'frame_tpr_fpr',name)
                
        if which_level is 'pixel_level':
            filename=os.path.join(self.save_path,'pixel_tpr_fpr',name)
        if not os.path.exists(filename):
                os.makedirs(filename)

        filename=filename+'/'+self.feature_name+'.txt'
        file = open(filename, 'w')
        file.close()
        feature_id=[dictFeatureName[self.feature_name+'+'+name] for i in range(len(all_fpr_value))]
        feature_id=np.array(feature_id).reshape(-1,1)
        np.savetxt(filename,np.c_[feature_id,np.array(all_tpr_value).reshape(-1,1),np.array(all_fpr_value).reshape(-1,1),np.array(thresholds_value).reshape(-1,1)],fmt="%d,%.5f,%.5f,%.5f")


        filename_all_cuboid=os.path.join(self.save_path,'all_cuboid',name)
        if not os.path.exists(filename_all_cuboid):
            os.makedirs(filename_all_cuboid) 
        filename_all_cuboid=filename_all_cuboid+'/'+self.feature_name+'.txt'
        file = open(filename_all_cuboid, 'w')
        file.close()
        np.savetxt(filename_all_cuboid,np.c_[self.test_data['timestamp'],np.array(cub_labels).reshape(-1,1),scores],fmt="%d,%d,%.5f")

        
    def compute_frame_tpr_fpr(self, pred_labels,flag):
        self.get_pred_results(pred_labels)

        timestamps,ind=np.unique(self.test_data['timestamp'], return_index=True)
        timestamps=timestamps[np.argsort(ind)]
        num_total = 2955

        i=0
        for timestamp in timestamps:
            idx=self.test_data['timestamp']==timestamp
            if i ==0:
                pred_results=(np.sum(self.test_data['pred_labels'][idx])>0.0)*1
            else:
                pred_results=np.append(pred_results,(np.sum(self.test_data['pred_labels'][idx])>0.0)*1)
            i+=1
        TP=np.sum(((pred_results+self.test_data['labels'])==2)*1)
        TN=np.sum(((pred_results+self.test_data['labels'])==0)*1)
        FP=np.sum(((pred_results-self.test_data['labels'])==1)*1)
        FN=np.sum(((pred_results-self.test_data['labels'])==-1)*1)

        if flag:
            omit_num = num_total - (TP+TN+FP+FN)
            TN = omit_num + TN
        else:
            omit_num = num_total - (TP+TN+FP+FN)
            FP = omit_num + FP

        TPR=TP/(TP+FN)
        FPR=FP/(FP+TN)
        return TPR, FPR

    def compute_pixel_tpr_fpr(self, pred_labels,flag):
        self.get_pred_results(pred_labels)

        timestamps,ind=np.unique(self.test_data['timestamp'], return_index=True)
        timestamps=timestamps[np.argsort(ind)]

        num_total = 2955
        i=0
        TP=0
        FP=0
        FN=0
        TN=0
        frame_idx = 0
        for timestamp in timestamps:
            idx=self.test_data['timestamp']==timestamp
            # frame_idx=np.abs(timestamp - self.test_data['pixel_labels_times'])<50000
            # gt_img = cv2.imread(self.test_data['pixel_labels_names'][frame_idx][0],0)

            gt_img = cv2.imread(self.test_data['pixel_labels_names'][0,frame_idx],0)
            frame_idx = frame_idx + 1

            if np.sum(self.test_data['pred_labels'][idx])>0.0 :
                
                patch_index = self.test_data['pred_labels'][idx]==1
                patch_pose = self.test_data['poses'][idx][patch_index]
                pred_img=np.zeros((260,346))
                for n in range(patch_pose.shape[0]):
                    pred_img[int(259-patch_pose[n,1]*14-14):int(259-patch_pose[n,1]*14),int(patch_pose[n,0]*18):int(patch_pose[n,0]*18+18)]=1

                iner_img = (gt_img+pred_img==256)*1
                if np.sum(gt_img/255)==0:
                    FP=FP+1
                else:
                    if np.sum(iner_img) > np.sum(gt_img/255) * 0.4:
                        TP=TP+1
                    else:
                        FP=FP+1
                        
            if i ==0:
                pred_results=(np.sum(self.test_data['pred_labels'][idx])>0.0)*1
            else:
                pred_results=np.append(pred_results,(np.sum(self.test_data['pred_labels'][idx])>0.0)*1)
            i+=1
        TN=np.sum(((pred_results+self.test_data['labels'])==0)*1)
        FN=np.sum(((pred_results-self.test_data['labels'])==-1)*1)
            

        if flag:
            omit_num = num_total - (TP+TN+FP+FN)
            TN = omit_num + TN
        else:
            omit_num = num_total - (TP+TN+FP+FN)
            FP = omit_num + FP

        TPR=TP/(TP+FN)
        FPR=FP/(FP+TN)
        return TPR, FPR


    # def compute_pixel_tpr_fpr(self, pred_labels,flag):
    #     self.get_pred_results(pred_labels)

    #     timestamps,ind=np.unique(self.test_data['timestamp'], return_index=True)
    #     timestamps=timestamps[np.argsort(ind)]
    #     num_total = 2955

    #     i=0
    #     for timestamp in timestamps:
    #         idx=self.test_data['timestamp']==timestamp
    #         if i ==0:
    #             pred_results=(np.sum(self.test_data['pred_labels'][idx])>0.0)*1
    #         else:
    #             pred_results=np.append(pred_results,(np.sum(self.test_data['pred_labels'][idx])>0.0)*1)
    #         i+=1
    #     # TP=np.sum(((pred_results+self.test_data['labels'])==2)*1)
    #     TN=np.sum(((pred_results+self.test_data['labels'])==0)*1)
    #     FP=np.sum(((pred_results-self.test_data['labels'])==1)*1)
    #     FN=np.sum(((pred_results-self.test_data['labels'])==-1)*1)


    #     TP=0
    #     FN_add = 0
    #     frame_idx = 0
    #     o_tp_idx = ((pred_results+self.test_data['labels'])==2)*1
    #     for timestamp in timestamps:
    #         if o_tp_idx[0,frame_idx] == 1 :
    #             gt_img = cv2.imread(self.test_data['pixel_labels_names'][0,frame_idx],0)

    #             patch_index = self.test_data['pred_labels'][idx]==1
    #             patch_pose = self.test_data['poses'][idx][patch_index]
    #             pred_img=np.zeros((260,346))
    #             for n in range(patch_pose.shape[0]):
    #                 pred_img[int(259-patch_pose[n,1]*14-14):int(259-patch_pose[n,1]*14),int(patch_pose[n,0]*18):int(patch_pose[n,0]*18+18)]=1

    #             iner_img = (gt_img+pred_img==256)*1
    #             if np.sum(iner_img) > np.sum(gt_img/255) * 0.4:
    #                 TP=TP+1
    #             else:
    #                 FN_add=FN_add+1
    #         frame_idx = frame_idx +1
    #     FN = FN + FN_add
    #     if flag:
    #         omit_num = num_total - (TP+TN+FP+FN)
    #         TN = omit_num + TN
    #     else:
    #         omit_num = num_total - (TP+TN+FP+FN)
    #         FP = omit_num + FP

    #     TPR=TP/(TP+FN)
    #     FPR=FP/(FP+TN)
    #     return TPR, FPR
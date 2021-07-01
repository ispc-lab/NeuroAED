from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os
from svm import OneClassSVM
from dictionary import Dictionary
from k_svd import ksvd
from sparse_autoencoder import SparseAutoencoder
from odl import OnlineDictionaryLearning
from utils import dump_model

class Classifiers(object):

    def __init__(self,dataset,feature_name,model_dir,k,hyperTune=True):
        self.construct_all_models(dataset,feature_name,model_dir,k,hyperTune)

    def construct_all_models(self,dataset,feature_name,model_dir,k,hyperTune):
        if hyperTune:
            #3 models KNN SCM and LR
            self.models={
                    'OneClassSVM':[OneClassSVM(dataset,'rbf', 0.1), 0],\
                    'SRC':[Dictionary(dataset,20,k), 0],\
                    'KSVD':[ksvd(dataset,20,k),0],\
 		    'ODL':[OnlineDictionaryLearning(dataset,k,100,100)],\
                        }
            for name,candidate_hyperParam in self.models.items():
                #update each classifier after training and tuning
                self.models[name] = self.train(candidate_hyperParam[0],feature_name,model_dir,name)
            print ('\nTraining process finished\n\n\n')
            
    
    def train(self,model,feature_name,model_dir,name):
        model.train()
        model_directory=os.path.join(model_dir)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory) 
	
        filename=os.path.join(model_directory,name+'_'+'train'+'_'+feature_name+'.pkl')
        dump_model(model,filename)
        
        return model

    def prediction_metrics(self,name,savePath=None):
        test_labels = self.models[name].test_labels
        scores = self.models[name].diag['test']['scores'][:,0]
        acc = self.models[name].diag['test']['acc'][0]
        auc = self.models[name].diag['test']['auc'][0]

        #accuracy
        print('{} test accuracy = {}\n'.format(name,acc))

        #AUC of ROC
        print('Classifier {} area under curve of ROC is {}\n'.format(name,auc))

        #ROC
        fpr, tpr, thresholds = roc_curve(test_labels, scores, pos_label=1)
        self.roc_plot(fpr,tpr,name,auc,savePath)

    def roc_plot(self,fpr,tpr,name,auc,savePath=None,showGraphic=True):
        plt.figure(figsize=(20,5))
        plt.plot(fpr,tpr)
        plt.ylim([0.0,1.0])
        plt.ylim([0.0, 1.0])
        plt.title('ROC of {}     AUC: {}\nPlease close it to continue'.format(name,auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        if savePath is not None:
            plt.savefig(os.path.join(savePath, name + '.png'))
        if showGraphic is True:
            plt.show()
             # plt.waitforbuttonpress()
            plt.pause(0.05)


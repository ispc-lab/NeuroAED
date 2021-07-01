#Abnormal behavior detection project
#main features: optical flow
#classifiers: SVM, LinearRegression, KNN
from Classifiers import *
# import cv2
import matplotlib.pyplot as plt
import glob
from ref_data.loadData import load_dataset
import os
from utils import result_plot,load_model
# from dictionary import Dictionary
# import mmcv
import json



dictKvalue={'TB1':40,'TB2':30,'TB3':50,\
                    'o2mv200_leijia':20,'Small Scale':30,'SB':30,\
                    'EMST':40,'Large Scale':40,'Single Scale':20,\
                     'STB':20,\
                    'Eight Orientation':20,\
                        'Two Orientation':20,\
                      }

def times2json(dicdata,filename):
    if isinstance(dicdata, dict):
        json_str = json.dumps(dicdata)
        with open(filename, 'w') as json_file:
            json_file.write(json_str)
    else:
        raise TypeError('invalid type of results')
   


def main ():
    directory='/xxxx/features'
    feature_names=[name for name in os.listdir(directory) if name == '200']
    feature_names.sort(reverse=True)
    for feature_name in feature_names:
        # K=dictKvalue[feature_name]
        K = 40
        data_dir=os.path.join(directory,feature_name)
        dataset = load_dataset('sigactcuboid',data_dir)
        classifiers = Classifiers(dataset.get_data(),dataset.feature_name,dataset.model_dir,K)
    

        for name,model in classifiers.models.items():#get each classifier
            model_directory=os.path.join(dataset.model_dir)
            modname = os.path.join(model_directory,name+'_'+'train'+'_'+feature_name+'.pkl')
            model_data = load_model(modname)
            labels,scores,test_time=model_data.predict()

            timefile = os.path.join(model_directory,name+'_'+'test'+'_'+feature_name+'.json')
            times2json(test_time,timefile)
            
            # dataset.get_tpr_fpr(labels,scores,'frame_level',name)
            # dataset.get_tpr_fpr(labels,scores,'pixel_level',name)

    
    
if __name__ == '__main__':
    main()

  

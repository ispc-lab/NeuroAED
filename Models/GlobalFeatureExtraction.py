# import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
import csv
import scipy.io
from sparse_autoencoder import SparseAutoencoder
import scipy.io as scio 

PATCH_SIZE = (18, 14)

def selected_patchs(frames, num_patch, patch_pose, n_feature):
    # Load the image. Here it is a 512*512*10 numpy array.
	numberOfFrames = len(frames)
	sizey, sizex = np.array(frames[0]).shape
	inputNodes = []
	pose_lines=0
	nframe=0
	for num_lines in num_patch:
		# for x in range(0, len(frames), n_feature):
		n_patch=num_lines
		for n in range(n_patch):
			patch_x_start = PATCH_SIZE[0] * patch_pose[pose_lines+n,0]
			patch_y_start = PATCH_SIZE[1] * patch_pose[pose_lines+n,1]
			for x in range(n_feature):
				node=[]
				for k in range(0,PATCH_SIZE[0]):
					for l in range(0, PATCH_SIZE[1]):
						intensity = frames[nframe*n_feature + x][patch_x_start+k,patch_y_start+ l]
						node.append(intensity)
				inputNodes.append(node)
		pose_lines+=n_patch
		nframe+=1
	
	inputNodes = np.array(inputNodes, dtype = np.float32)  #Convert to np array
	inputNodes = np.transpose(inputNodes)
	inputNodes = (inputNodes - inputNodes.mean(axis=0)) / inputNodes.std(axis=0) #Normalization,column
	
	return inputNodes
	


def readFrames(directory, which_set, timestamps, patch_pose):
	videos = []
	
	videos = [name for name in os.listdir(directory,which_set)]
	videos.sort()

	num_patch=np.zeros((len(np.unique(timestamps)),1))
	i=0
	for timestamp in timestamps:
		num_patch[i,0]=np.sum((timestamps==timestamp)*1)
		i+=1


	for video in videos:
		frames = []
		
		orientationFileNames = [os.path.join(directory,video,'Orientation', name) for name in os.listdir(os.path.join(directory,video,'Orientation'))]
		tempGradientFileNames = [os.path.join(directory,video,'TempGradient', name) for name in os.listdir(os.path.join(directory,video,'TempGradient'))]
		timeStampsFileNames = [os.path.join(directory,video,'TimeStamps', name) for name in os.listdir(os.path.join(directory,video,'TimeStamps'))]
		orientationFileNames.sort()
		tempGradientFileNames.sort()
		timeStampsFileNames.sort()

		for i in range(len(orientationFileNames)):
			all_orientation = scio.loadmat(orientationFileNames[i])['filteredOrientationFrame']
			all_ox = scio.loadmat(tempGradientFileNames[i])['ox']
			all_oy = scio.loadmat(tempGradientFileNames[i])['oy']
			all_frequency = scio.loadmat(tempGradientFileNames[i])['posLastEventPosition']
			all_postimestamp = scio.loadmat(timeStampsFileNames[i])['posTimeStamp']	

			frames.append(all_orientation)
			frames.append(all_ox)
			frames.append(all_oy)
			frames.append(all_frequency)
			frames.append(all_postimestamp)

		n_feature=5
		
		inputNodes=selected_patchs(frames, num_patch, patch_pose, n_feature)
		input_size = inputNodes.shape[0]
		hidden_size = inputNodes/n_feature
		if which_set is 'train':
			model=SparseAutoencoder(input_size,hidden_size,'Dining')
			globalD=model.train(inputNodes)
		if which_set is 'test':
			model=SparseAutoencoder(input_size,hidden_size,'Dining')
			globalD=model.predict(inputNodes)
	return inputNodes

# if __name__ == "__main__":
# 	which_set = 'train'
# 	dir = 'UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'
# 	readFrames(dir,which_set)



					



	


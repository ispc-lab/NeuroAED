# NeuroAED: Towards Efficient Abnormal Event Detection in Visual Surveillance with Neuromorphic Vision Sensor
This is the NeuroAED dataset and implementation code for the following paper. If you use any of them, please cite: 

# NeuroAED dataset
Considering the lack of a neuromorphic benchmark for the abnormal event detection, we record the first neuromorphic vision based abnormal event detection dataset and make it available to the research community at the link:
![image](https://github.com/ispc-lab/NeuroAED/tree/master/images/Dataset_samples.png)
![image](https://github.com/ispc-lab/NeuroAED/tree/master/images/Dataset_description.png)
The NeuroAED dataset comprises 152 samples of four different indoor and outdoor scenarios, and is split into four sub-dataset: Walking, Campus, Square and Stair dataset. And each dataset contains two slice sequences: training samples and testing samples. The training samples only contain normal events, while testing samples are both normal and abnormal events. For each slice sample of the NeuroAED dataset, the groundtruth annotation of a binary flag indicating normal or abnormal events occur is provided. With the exception of the Square dataset, the manually generated pixel-level binary masks are contained in each slice sample, which identify the abnormal events regions.

# NeuroAED: Towards Efficient Abnormal Event Detection in Visual Surveillance with Neuromorphic Vision Sensor
This is the NeuroAED dataset and implementation code for the following paper. If you use any of them, please cite: 
```
@ARTICLE{,
        title = {NeuroAED: Towards Efficient Abnormal Event Detection in Visual Surveillance with Neuromorphic Vision Sensor},
      journal = {},
         year = {2020},
        pages = {},
      author  = {}, 
       eprint = {} 
}
```
# NeuroAED dataset
Considering the lack of a neuromorphic benchmark for the abnormal event detection, we record the first neuromorphic vision based abnormal event detection dataset and make it available to the research community at the link:

![image](https://github.com/ispc-lab/NeuroAED/blob/master/images/Dataset_description.png)


![image](https://github.com/ispc-lab/NeuroAED/blob/master/images/Dataset_samples.png) 

The NeuroAED dataset comprises 152 samples of four different indoor and outdoor scenarios, and is split into four sub-dataset: Walking, Campus, Square and Stair dataset. And each dataset contains two slice sequences: training samples and testing samples. The training samples only contain normal events, while testing samples are both normal and abnormal events. For each slice sample of the NeuroAED dataset, the groundtruth annotation of a binary flag indicating normal or abnormal events occur is provided. With the exception of the Square dataset, the manually generated pixel-level binary masks are contained in each slice sample, which identify the abnormal events regions.

# Framework of NeuroAED
We extract the optical flow information from training sample and select activated event cuboids based on the optical flow and event density to locate foreground. For each activated event cuboid, the proposed event-based multiscale spatio-temporal (EMST) descriptor is extracted and feed into models to learn the normal patterns. The trained models are used to identify descriptors of abnormal patterns extracted from the testing sample.
![image](https://github.com/ispc-lab/NeuroAED/blob/master/images/Dataset_samples.png)

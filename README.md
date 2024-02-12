# CSE151A_Project
## Abstract
In the cutting-edge field of AI-powered home robots, our proposed development of a beer-serving home robot with semantic navigation within indoor spaces is definitely innovative. Our robot, powered by custom-trained machine learning models via a suite of sensors including LIDAR, will be able to discern and interact with various elements of a household environment, such as identifying a kitchen or navigating around furniture. We propose that we shall train a convolutional neural network to accomplish the task of classifying images of indoor spaces by room name. Additionally, we also propose using room layout information derived from LIDAR as an input to our CNN. This, in aggregation with classical navigation techniques such as SLAM, will allow us to establish a mapping of 3d coordinates to semantic space (room name), making possible our end goal of a robot which is capable of semantic navigation given a list of high-level unstructured waypoints.

## MILESTONE 2
In this assignment you will need to:

We each created/shared our GitHub Accounts: 
- AriYoung00
- abenstirling
- crae6
- mBookUCSD
- AzbijariKeyan
- alanm319
- kdharanipathi

Our Github repo is found here: https://github.com/abenstirling/CSE151A_Project

Perform the data exploration step (i.e. evaluate your data, # of observations, details about your data distributions, scales, missing data, column descriptions) Note: For image data you can still describe your data by the number of classes, # of images, plot example classes of the image, size of image. 
 
Are sizes uniform? 
- No, images in the dataset are a range of sizes. 

Do they need to be cropped? 
- Yes. Some of our depth will be “fake” model output and will result in different dimensions and absolute values. 

Normalized?
- Yes. All depth maps will be expressed in meters from the camera at each pixel.

Dataset (With original data if we can get it): 
- https://www.kaggle.com/datasets/itsahmad/indoor-scenes-cvpr-2019/data

Depth Estimation for the existing dataset with glpn-nyu:
- https://huggingface.co/vinvino02/glpn-nyu

Plot your data. For image data, you will need to plot your example classes.

![clases](https://github.com/abenstirling/CSE151A_Project/blob/main/Notebooks/classes.png?raw=true)

How will you preprocess your data? 

Aside from cropping, and normalization, we are using glpn-nyu and putting in a new depth dimension to our image dataset. 
![test](https://github.com/abenstirling/CSE151A_Project/blob/main/Notebooks/DepthEstimation.png?raw=true)

- [X] Your jupyter notebook(s) should be uploaded to your repo.
- [X] https://github.com/abenstirling/CSE151A_Project/blob/main/Notebooks/

Jupyter Notebook data download and environment setup requirements: 
- [X] !wget !unzip like functions as well as !pip install functions for non standard libraries not available in colab are required to be in the top section of your jupyter lab notebook.
- [X] Having the data on GitHub (you will need the academic license for GitHub to do this)


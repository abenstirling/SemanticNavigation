# CSE151A_Project
## Abstract
In the cutting-edge field of AI-powered home robots, our proposed development of a beer-serving home robot with semantic navigation within indoor spaces is definitely innovative. Our robot, powered by custom-trained machine learning models via a suite of sensors including LIDAR, will be able to discern and interact with various elements of a household environment, such as identifying a kitchen or navigating around furniture. We propose that we shall train a convolutional neural network to accomplish the task of classifying images of indoor spaces by room name. Additionally, we also propose using room layout information derived from LIDAR as an input to our CNN. This, in aggregation with classical navigation techniques such as SLAM, will allow us to establish a mapping of 3d coordinates to semantic space (room name), making possible our end goal of a robot which is capable of semantic navigation given a list of high-level unstructured waypoints.

## MILESTONE 3
Our First Model Notebook: https://github.com/abenstirling/CSE151A_Project/blob/main/Models/First_Model.ipynb

Comparing training vs test error. Is our model overfitting or underfitting? 
Fitting Graph and where our model fits

Next Two Models: If we are overfitting, Since our model is overfitting we can improve performance by augmenting our data further. We can increase the diversity of the training set by rotating, scaling, and flipping images. Dropout layers also reduce dependency on specific features and promote generalization. We can also apply L1 or L2 regularization to the weights of the network to penalize large weights. 

 If we are underfitting, Increasing the depth of our network with additional convolutional layers or neurons in the dense layers will allow us the capture more complex features. 

 General improvements: To improve general performance we can do some hyperparameter tuning to experiment with different learning rates, batch sizes, and optimizer settings to find the best possible configuration for our model. We can add batch normalization layers to stabilize and speed up training. 

 If we decide to use only a subset of our dataset since it is so large, we can include that here for improvements.

Conclusion of our First Model: This first model is a great starting point and a step in the right direction. It helps us understand how our current setup and data prep methods are doing. By carefully fixing any problems we find, we can gradually make our next two models better by not making the same mistakes we made here. We aim to build a strong model that can correctly identify different rooms in a house, making our beer-delivering home robot smarter and more capable of moving around on its own. Trying new things and making adjustments are crucial steps to improve our next models.  


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


# CSE151A_Project
# Abstract
In the cutting-edge field of AI-powered home robots, our proposed development of a beer-serving home robot with semantic navigation within indoor spaces is definitely innovative. Our robot, powered by custom-trained machine learning models via a suite of sensors including LIDAR, will be able to discern and interact with various elements of a household environment, such as identifying a kitchen or navigating around furniture. We propose that we shall train a convolutional neural network to accomplish the task of classifying images of indoor spaces by room name. Additionally, we also propose using room layout information derived from LIDAR as an input to our CNN. This, in aggregation with classical navigation techniques such as SLAM, will allow us to establish a mapping of 3d coordinates to semantic space (room name), making possible our end goal of a robot which is capable of semantic navigation given a list of high-level unstructured waypoints.
# MILESTONE 4
Milestone: Building and Evaluating the Second Model

In this milestone, you will focus on building your second model. You will also need to evaluate this model and see where it fits in the underfitting/overfitting graph.

### 1. Evaluate your data, labels and loss function

![test](https://github.com/abenstirling/CSE151A_Project/blob/main/model2_accuracy.png?raw=true)
![test](https://github.com/abenstirling/CSE151A_Project/blob/main/model2_loss.png?raw=true)

We attempted to remove our depth from the x variable, and did not have much success with using image net. Then we pivoted to a full-blown custom CNN, which is totally different from our first model which was a Sequential one.

- **Data**: Ensure the dataset has sufficient quality, diversity, and representativeness for the classification task, and consider preprocessing and augmentation techniques.
- **Labels**: Verify the accuracy and consistency of labels across the dataset, handle class imbalances if present, and use appropriate encoding for multi-class classification.
- **Loss function**: Choose a loss function that aligns with the problem type and desired optimization objective.

### 2. Train your second model ✅

- Models/Second_model.ipynb

### 3. Evaluate your model compare training vs test error

The training error can be inferred from the training loss, which reaches a value of around 1.0. This indicates that the model has learned to fit the training data relatively well. The test error, represented by the validation loss, is higher than the training error, with a value around 1.5. This suggests that the model's performance on unseen data is not as good as its performance on the training data. The higher test error compared to the training error is potentially overfitting, we are unsure though.

### 4. Where does your model fit in the fitting graph, how does it compare to your first model?

We don't have any clear signs of overfitting or underfitting which is great. The graphs of our accuracy and loss also look much better compared to our first model. After around 50 epochs our loss and accuracy steady out at a good rate and a healthy distance is established between the train data and unseen testing data. This is a feature of a well-trained model and confirms that our model seems to be well-fit.

### 5. Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?

Performed data augmentation by flipping, shifting, and rotating our training images. We found that we got better results without this, but we will attempt again in our third model. Also, batch normalization!

### 6. What is the plan for the next model you are thinking of and why?

We are going to attempt to improve our model using a convolutional based encoding layer that is specifically designed to deal with depth information differently because we're currently treating the depth data as just one other dimension rather than an inherently different data type which it really is. This involves splitting the data streams into multiple parts and interlinking them rather than just throwing a four-dimensional image into a standard image processing technique.

### What is the conclusion of your 2nd model?

In conclusion, our second machine learning model showed substantial improvements over our initial model in terms of both accuracy and loss metrics.

The training and validation accuracy of the second model reached approximately 70%, a significant increase from the roughly 50% accuracy achieved by the first model. This 20 percentage point gain in accuracy indicates that the second model is considerably better at correctly predicting outcomes on the training and validation datasets.

In addition, the second model was able to reduce the loss to less than 1.5, an improvement over the 1.6 loss of the initial model. A lower loss value signifies that the second model's predictions are closer to the actual target values on average.

### What can be done to possibly improve it?

There are several potential avenues for further enhancing the performance of the second model:

- Conducting more extensive hyperparameter tuning through techniques like grid search or random search to find an optimal combination of model hyperparameters
- Increasing the size and diversity of the training dataset to improve the model's ability to generalize to new, unseen data

### How did it perform compared to your first and why?

Compared to the first model, the second model's 70% accuracy versus 50% and <1.5 loss versus 1.6 represent substantial improvements in predictive power and optimization of the loss function. The better performance is likely due to some combination of higher quality input data, a more suitable model architecture, and better tuned hyperparameters. However, without knowing the specific differences between the two models, it's difficult to pinpoint the exact reasons for the performance gap.




# MILESTONE 3
Our First Model Notebook: https://github.com/abenstirling/CSE151A_Project/blob/main/Models/First_Model.ipynb

![Test](https://github.com/abenstirling/CSE151A_Project/blob/main/accuracy.png?raw=true)
![Test](https://github.com/abenstirling/CSE151A_Project/blob/main/loss.png?raw=true)

Comparing training vs test error. Is our model overfitting or underfitting? 
We are not sure, though we don't see an overfitting spike and it seems that the difference between test and training error is beginning to diverge. Hence, we believe that our current model is underfitting. 

Next Two Models: Since our model is underfitting we can improve performance by augmenting our data further. We can increase the diversity of the training set by rotating, scaling, and flipping images. Dropout layers also reduce dependency on specific features and promote generalization. We can also apply L1 or L2 regularization to the weights of the network to penalize large weights. 

General improvements: To improve general performance we can do some hyperparameter tuning to experiment with different learning rates, batch sizes, and optimizer settings to find the best possible configuration for our model. We can add batch normalization layers to stabilize and speed up training. 

 If we decide to use only a subset of our dataset since it is so large, we can include that here for improvements.

Conclusion of our First Model: This first model is a great starting point and a step in the right direction. It helps us understand how our current setup and data prep methods are doing. By carefully fixing any problems we find, we can gradually make our next two models better by not making the same mistakes we made here. We aim to build a strong model that can correctly identify different rooms in a house, making our beer-delivering home robot smarter and more capable of moving around on its own. Trying new things and making adjustments are crucial steps to improve our next models.  


# MILESTONE 2
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


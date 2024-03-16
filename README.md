# CSE151A_Project

# Abstract
In the cutting-edge field of AI-powered home robots, our proposed development of a beer-serving home robot with semantic navigation within indoor spaces is definitely innovative. Our robot, powered by custom-trained machine learning models via a suite of sensors including LIDAR, will be able to discern and interact with various elements of a household environment, such as identifying a kitchen or navigating around furniture. We propose that we shall train a convolutional neural network to accomplish the task of classifying images of indoor spaces by room name. Additionally, we also propose using room layout information derived from LIDAR as an input to our CNN. This, in aggregation with classical navigation techniques such as SLAM, will allow us to establish a mapping of 3d coordinates to semantic space (room name), making possible our end goal of a robot which is capable of semantic navigation given a list of high-level unstructured waypoints.

# Introduction
### Project Overview
In the field of artificial intelligence (AI) and robotics, the development of home robots that can interact with their environment represents a significant step forward. This paper discusses the development of a new kind of home robot, a robot designed to serve beer, which can navigate through many different indoor spaces. Our project leverages custom-trained machine learning models, supported by a suite of sensors including LIDAR, to enable the robot to recognize and navigate through different parts of a home, such as kitchens and living rooms, and to maneuver around obstacles like furniture. 
The focus of our project is to train a convolutional neural network (CNN) to identify indoor spaces by room name using images, with the addition of spatial data from LIDAR to improve the model’s performance. 
We emphasize the functionality of delivering beverages and the technological innovation of having a robot to understand and navigate through indoor environments more efficiently and effectively. By combining AI with precise sensor data we can push the boundaries of current home robot capabilities.

### Motivation and Significance
Our motivation for doing this project is that while there are large data sets of images of indoor spaces that are tagged well, very little exists for images with depth information to solve this problem and our goal is to supplement these types of models with data from general depth models. The significance of this is that it will allow depth cameras to be used for this task better than if they were only using their RGB data and not using their depth sensors. 

### Repository Structure
The file structure of our project is the following: 
```
/Data
  Prepare_data.sh
/Models
  First_Model.ipynb
  Second_Model.ipynb
  Third_Model.ipynb
/Notebooks
  Add_Depth_CVPR.ipynb
  GetDatasetImageInfo.ipynb
  Im
```


# Methods
![Test](https://github.com/abenstirling/CSE151A_Project/blob/6b04426a3301f522dc857a663ee3a93c76651ac6/model_construction.jpg?raw=true)
### Data Exploration
The dataset used is the MIT Indoor Scenes dataset, which is a collection of about 15620 images from 67 scene categories, where each category contains a minimum of 100 images. The 67 scenes are further organized in 5 “scene groups”: Store, Home, Public Spaces, Leisure, and Working Place. Images are in jpg format and come in a range of different sizes and resolutions, where the minimum resolution is 200 pixels wide. Images do not contain any depth data, which is another dimension that our model depends on and was added during preprocessing. Only the Home scene group is used for training and validation of the model. The dataset was then further refined to remove the scene categories “winecellar”, “lobby”, “closet”, and “staircase”.
### Data Preprocessing
The first step of our data preprocessing was to use our depth generation model to create depth images of all of our training data. We then proceeded to crop and resize our image data and depth data to 64x36. This is done within the ImageSizing.ipynb file. Next, we store our image files as numpy arrays by converting them to store each pixel as a value between 0 and 255 (RGB scale). Then, we scale the RGB values of each image to between 1 and 0 by dividing each numpy array by 255. Next, we resize our data to 64x64 and add the depth dimension to the data by concatenating the image and depth data together. This ‘final’ data is then stored as the X value and it’s classification is stored as the y value. Finally, we are able to train-test split our data and finish preprocessing. 
### Model 1
```python
model = Sequential([
    InputLayer(input_shape=(64, 64, 4)), # Adjust the input shape to match the input of the images we decide on (e.g. 64 x 64 pixels)
    Conv2D(64, kernel_size=(4, 4), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=L1L2(0.0001)),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
```
This model consists of multiple convolutional layers followed by max pooling layers and dense dropouts within the fully connected layers. 
### Model 2
```python
input_shape = (64, 64, 4)

# we can create an input tensor here 
inputs = Input(shape=input_shape)

# our custom CNN architecture, we got the batch normalization!!
x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x) 
model = Model(inputs=inputs, outputs=output)
```
The model consists of multiple convolutional layers with batch normalization, max pooling, dropout, and dense layers, and is trained using data generators with a learning rate scheduler callback to reduce the learning rate when the validation loss plateaus.

### Model 3
```python
rgb_input = Input(shape=(64, 64, 3))
depth_input = Input(shape=(64, 64, 1))

# RGB layers
rgb_stream = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l1(0.001), strides=(1,1))(rgb_input)
rgb_stream = BatchNormalization()(rgb_stream)
rgb_stream = MaxPooling2D(pool_size=(2, 2))(rgb_stream)
rgb_stream = Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(0.001), strides=(1,1))(rgb_stream)
rgb_stream = BatchNormalization()(rgb_stream)
rgb_stream = MaxPooling2D(pool_size=(2, 2))(rgb_stream)
rgb_stream = Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(0.001), strides=(1,1))(rgb_stream)
rgb_stream = BatchNormalization()(rgb_stream)
rgb_stream = MaxPooling2D(pool_size=(2, 2))(rgb_stream)

# Depth layers
depth_stream = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l1(0.001), strides=(1,1))(depth_input)
depth_stream = BatchNormalization()(depth_stream)
depth_stream = MaxPooling2D(pool_size=(2, 2))(depth_stream)
depth_stream = Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(0.001), strides=(1,1))(depth_stream)
depth_stream = BatchNormalization()(depth_stream)
depth_stream = MaxPooling2D(pool_size=(2, 2))(depth_stream)
depth_stream = Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(0.001), strides=(1,1))(depth_stream)
depth_stream = BatchNormalization()(depth_stream)
depth_stream = MaxPooling2D(pool_size=(2, 2))(depth_stream)

combined = Concatenate()([rgb_stream, depth_stream])
combined = Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same', kernel_regularizer=l2(0.001), strides=(1, 1))(combined)
combined = Dropout(0.3)(combined)
combined = BatchNormalization()(combined)
combined = GlobalAveragePooling2D()(combined)
combined = Dropout(0.25)(combined)
combined = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
combined = Dropout(0.2)(combined)
combined = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
combined = Dropout(0.2)(combined)
combined = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)

output_layer = Dense(10, activation='softmax')(combined)
```
In this model, we initially separate the model into two streams of layers which separately process the RGB and depth data. We chose to do this because these two types of data communicated fundamentally different things, and are also described with different units -- RGB is normalized, whereas depth values are not. The separate streams consist of a sequence of `Conv2D` layers. After this they are concatenated, before we stack several dense layers in front of the output layer, interspersed with droupouts.

# Results
### Data Exploration
After refining our data set, the dataset was shrunk down from 15620 images to 3672 images in total. The remaining scene categories were bedroom, bathroom, kitchen, living room, laundry, garage, nursery, pantry, children room, dining room, and corridor. 
### Model 1
Our first model was underfitting, but it was a good starting step for our project. Our model was performing with a training loss of around 1.6 and a validation loss of around 1.7. As for our accuracy, the training accuracy was around 52% and the validation accuracy was around 47%. This first model really helped us understand the current setup and our data prep. 

![Test](https://github.com/abenstirling/CSE151A_Project/blob/main/accuracy.png?raw=true)
![Test](https://github.com/abenstirling/CSE151A_Project/blob/main/loss.png?raw=true)

### Model 2
Our second machine learning model showed substantial improvements over our initial model in terms of both accuracy and loss metrics. The training and validation accuracy of the second model reached approximately 70%, a significant increase from the roughly 50% accuracy achieved by the first model. This 20 percentage point gain in accuracy indicates that the second model is considerably better at correctly predicting outcomes on the training and validation datasets. In addition, the second model was able to reduce the loss to less than 1.5, an improvement over the 1.6 loss of the initial model. A lower loss value signifies that the second model's predictions are closer to the actual target values on average. Compared to the first model, the second model's 70% accuracy versus 50% and <1.5 loss versus 1.6 represent substantial improvements in predictive power and optimization of the loss function.
![test](https://github.com/abenstirling/CSE151A_Project/blob/main/model2_accuracy.png?raw=true)
![test](https://github.com/abenstirling/CSE151A_Project/blob/main/model2_loss.png?raw=true)
### Model 3
Our third model, unfortunately, did not show substantial improvement over our second model in terms of either accuracy or loss. Although we believed that the new stream-based architecture would yield better results as a result of specializing layers to the type of information they operate on, we were unable to tune this model any further than a training accuracy of 0.73 and a validation accuracy of 0.65. We see a test accuracy of 0.65 as well, so we do not believe that this model is overfitting to the training data. Although this model initially had some issues with overfitting, we managed to combat these issues using dropout layers. Evidence indicates that we are also not underfitting, because neither decreasing the dropout rate nor removing dropout layers improves validation accuracy. It is out opinion that the next step towards improving this model would be to incorporate a greater volume of training data.
![test](https://github.com/abenstirling/CSE151A_Project/blob/main/model3_accuracy.png?raw=true)
![test](https://github.com/abenstirling/CSE151A_Project/blob/main/model3_loss.png?raw=true)

# Discussion
### Data Exploration
The MIT Indoor Scenes Dataset was chosen because it was fairly large and comprehensive, and it perfectly fit our task of classifying rooms. The dataset was reduced because it also contained large number of non-home scenes, and our task is only concerned with classifying rooms in a typical home. Other catagories in the "home" scene group such as "wineceller" and "lobby" were also removed as they do not reflect rooms one might find in a typical American home. Although the dataset was dramatically reduced, we were still left ~4000 images, which was sufficient for training our models as demonstrated.
### Preprocessing
Images in the dataset were not uniform; they varied drastically in size and aspect ratio. Additionally, they lacked depth, which is an important attribute that our model depends on. So the first step in preprocessing was to obtain depth maps of each scene using the GPLN depth model. The raw images were resized and their RGB values normalized to improve the performance and efficiency of our models. This is because uniformity and normalization in the RGB data helps speed up the convergence of the gradient descent in the training process and increases generalization of the model.
### Model 1
For our first model, we constructed a fairly typical CNN image classifier. The network is starts with a cascade of convolutional layers combined with Pooling layers. Convolutional layers are essential to image classification to create feature maps using filters. These convolutional layers are followed by Pooling layers to reduce the spacial dimensions of the output of each convolutional layer, which reduces the training time and overfitting. A flatten layer is added after the last convolutional layer to flatten the input so that it can be fed into the first Dense layer. We also utilized dropout layers in between some of the dense layers to perform regularization and reduce the chance of overfitting. As shown in the loss graph of the model, the divergence of the training loss and validation loss indicates that our model is underfitting and is therefor lacking in complexity of architecture for the task at hand. In the end, the model performed fairly well for a first attempt at a model.
### Model 2
We improved the architecture of the model drastically by incorporating batch normalization, a global average Pooling layer, more L2 regularization, and higher dropout layers. Adding a batch normalization layer after every convolutional layer was added as a regularization technique to prevent overfitting and also to speed up training. The flatten layer was replaced with  Global Average Pooling layer, which should reduce the need for many dense layers and make our model more generalizable, further preventing overfitting.
In addition to this restructuring our model architecture, we performed some data augmentation such flipping, shifting, and rotating our training images. This model outperformed the our first by a wide margin. The better performance of the second model over the first is likely due to some combination of higher-quality input data, a more suitable model architecture, and better-tuned hyperparameters.
### Model 3
For our third model, we took a novel approach by creating a dual-stream architecture that processes the RGB and depth information separately before combining them. This allows the model to learn features specific to each modality and then fuse them together for the final classification. The RGB stream and depth stream each consist of a series of convolutional layers with batch normalization and max pooling. The convolutional layers extract features from the input images, while the batch normalization helps regularize the model and speed up training. Max pooling reduces the spatial dimensions, which helps prevent overfitting and reduces computational complexity.

After processing the RGB and depth streams separately, we concatenate their outputs and pass them through additional convolutional, dropout, and dense layers, reducing the risk of overfitting. We also incorporated L1 and L2 regularization in the convolutional and dense layers to add additional constraints on the model's weights, further reducing the risk of overfitting. This dual-stream architecture allows the model to effectively combine the information from both the RGB and depth modalities, potentially leading to improved performance compared to processing them together from the start. The combination of regularization techniques (batch normalization, dropout, L1/L2 regularization) should help the model generalize well to new data. Although we did not see much improvement with this model as compared to Model 2, we stil believe the architecture to be fundamentally superior. With more data, we would expect the performance of this model to improve dramatically, given that the architecture should generalize to new data better than our previous architectures did.
# Conclusion
In this project, we developed a series of CNN models to classify indoor scenes for a beer-serving home robot using the MIT Indoor Scenes Dataset. Our goal was to create a model that could accurately classify images of rooms in a typical home, which would enable our robot to navigate semantically through an indoor environment.

Through the course of this project, we explored the dataset, preprocessed the images, and developed three distinct CNN architectures. The first model, although simple, provided a good starting point and helped us understand the task at hand. The second model incorporated several improvements, such as batch normalization, global average pooling, and data augmentation, which significantly boosted its performance over the first model.

The third model introduced a novel dual-stream architecture that separately processed the RGB and depth information before combining them. This approach allowed the model to learn features specific to each modality and then fuse them for the final classification. Although this model did not show substantial improvement over the second model, we believe that with more training data, its performance could be dramatically enhanced due to its architecture that should generalize well to new data.

Throughout the development process, we employed various regularization techniques, such as dropout, L1/L2 regularization, and batch normalization, to prevent overfitting and improve the models' ability to generalize to new data.

In conclusion, our project demonstrates the potential of using CNNs to classify indoor scenes for semantic navigation in home robots. The models we developed, particularly the second and third architectures, show promising results and provide a solid foundation for further research and development in this field. With additional data and fine-tuning, we believe that our approach could be extended to create even more accurate and robust models for indoor scene classification, bringing us closer to the goal of developing intelligent, semantically-aware home robots.

# Collaboration
Ben Stirling - 4th year computer engineering student - I helped design and implement model 2. 

Alan Mohamad - 4th year computer engineering student - I helped with writing the report and sending out when2meet to organize meetings.

Moshe Bookstein - 3rd year computer science student - I helped do initial data exploration and helped do tuning for model 1 and implemented the architecture for model 3 as well as initial exploration of model 3, helped write and format reports.

Keyan Azbijari - 4th year Math-CS student- I helped with writing the ReadME/report and facilitating meetings.

Kruti Dharanipathi - 3rd year Computer Engineering Student- I helped with writing the reports, feedback, and meeting facilitation.

Ariel Young - 4th year Computer Engineering Student - I worked on data preprocessing and ensuring that everyone could get setup and run the .ipynb locally, and helped to tune models 1 and 3 to a reasonably degree of accuracy and low loss. I also helped to write the report.

Carson Rae - 4th-year Computer Engineering Student - I worked on data preprocessing (adding depth dimension, normalization, one-hot encoding, etc.) and designed and implemented the first model, along with initial hyperparameter tuning.

# Previous Milestones:
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


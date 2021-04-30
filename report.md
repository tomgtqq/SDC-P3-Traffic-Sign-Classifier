# **Traffic Sign Recognition** 
---
The project is a deep neural and convolutional neural networks to calssify traffic signs.

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./src/Visualization.jpg "Visualization"
[image2]: ./src/leNet.jpg "LeNet"
[image3]: ./src/train_supervise_learning.png "Train Fine-tuning"
[image4]: ./src/result.jpg "result"
[image5]: ./src/new_traffic_signs.jpg "Traffic Signs"
[image6]: ./src/placeholder.png "Traffic Sign 3"
[image7]: ./src/placeholder.png "Traffic Sign 4"
[image8]: ./src/placeholder.png "Traffic Sign 5"

You're reading it! and here is a link to my [project code](https://github.com/tomgtqq/SDC-P3-Traffic-Sign-Classifier)

### Data Set Summary & Exploration

### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

### 1. Pre-processing the image data. 

As a first step, I decided to normalize the images to 0~1, then I decided to generate additional data, So I augmented images with rotation, width shift, and heights shift, horizontal flip.


### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model. 

I builded 2 network as final model:

##### LeNet with additional 2 layers to improve network ability.

![alt text][image2]
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Conv2D  	| filters=6, kernel size=(5,5),  strides=(1,1),  activation='relu',padding='valid'|
|MaxPooling2d|	pool_size=(2, 2), strides=(1, 1), padding='valid'|
| Conv2D  	| filters=16, kernel size=(5,5),  strides=(1,1),  activation='relu',padding='valid'|
|MaxPooling2d|	pool_size=(2, 2), strides=(1, 1), padding='valid'|
| Conv2D  	| filters=32, kernel size=(5,5),  strides=(1,1),  activation='relu',padding='valid'|
|MaxPooling2d|	pool_size=(2, 2), strides=(1, 1), padding='valid'|
| Conv2D  	| filters=64, kernel size=(5,5),  strides=(1,1),  activation='relu',padding='valid'|
|MaxPooling2d|	pool_size=(2, 2), strides=(1, 1), padding='valid'|
| Conv2D  	| filters=128, kernel size=(5,5),  strides=(1,1),  activation='relu',padding='valid'|
|MaxPooling2d|	pool_size=(2, 2), strides=(1, 1), padding='valid'|
| Flatten				|      									|
|Dense(120)						|		activation='relu', kernel_regularizer='l2'								|
|	Dense(84)						|	activation='relu',kernel_regularizer='l2'												|
|	Dense(43)						|												| 


---


 #### Transfer Learning .
I used a pre-training  MobileNet model. and fine-tuning train all parameters. The result is so exciting for this task.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Conv2D  	| filters=6, kernel size=(5,5),  strides=(1,1),  activation='relu',padding='valid'|
|	Dense(43)						|												| 


### 3. Trained the model. 

To train the model, I recorded the whole training process as following. 

### Record

|   Model    | layers|BATCH_SIZE| LEARNING_RATE |Optimizer|Training loss    |   Training Accuracy    | Validation loss| Validation Accuracy   | Analyze | Improve | Save Model|
|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------------|-------------|-------------|
|LeNet| raw |64 | 0.001|adam|0.8954|0.7827    |  0.9558  |  0.7878      | High Biase         |  Bigger network |--|
|LeNet| +Conv2D(64) +MaxPooling2D  |64 | 0.001|adam|0.5518|0.8769    |  0.3953  |  0.9308      | High Biase         |  Bigger network |--|
|LeNet| +Conv2D(64) +MaxPooling2D +Conv2D(128) +MaxPooling2D |64 | 0.001|adam|0.4543|0.8769    |  0.2562  |  0.9460      | Met Solution Approach       |  --  |./save_model/LeNet/1619628590|
|[mobilenet_v2](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4)| +Dense(42) |64 | 0.001|adam|0.9566|0.9024    |  0.7531        |0.7932     | Overfitting on train set   |Train hub model| --|
|mobilenet_v2| trainable=True |64 | 0.001|adam|0.1285|0.9930    | 0.1864      |0.9764 | Met Solution Approach | -- |./save_model/mobilenet/1619661015| 

When the model is High biase, I tried to add more Conv2D layers and train longer. If the model is Overfitting on the train set. I tried to set "L2 regularization" and  "EarlyStopping". 

![alt text][image3]


### 4. Approach solution

My final model results were:
* Training set accuracy of 88% , using Data with aumentation 
* Validation set accuracy of  94%
* Test set accuracy of  91%

![alt text][image4]


If an iterative approach was chosen:
I think the important design is to try a small network at first. then add more layers to improve network ability.
I would choose some pre-train models to solve this task. So I choose the mobileNet model with light weights. if the mobileNet didn't meet the approach. I would choose ResNet and so on.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web

Here are five German traffic signs that I found on the web:

 ![alt text][image5] 

I used first model to predict the images.

```
predictions = model.predict_classes(test_images)
m = tf.keras.metrics.Accuracy()
m.update_state(predictions, test_labels)
print("The accuracy {:2.0f} %".format(m.result().numpy()*100) )

The accuracy 100 %
```

It's so exciting result, and I had to take so long time to train and fine-tuning hyperparameters. such as Bach Size and learning rate. But I figure out that Model Architecture and Data with augmentation are key for this task.

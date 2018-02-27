
# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center_2016_12_01_13_39_29_951.jpg "Center Image"
[image3]: ./examples/center_2016_12_01_13_39_29_951.jpg "Normal Image"
[image4]: ./examples/center_2016_12_01_13_39_29_951_flip.jpg "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run.mp4 for testing results.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
The model architecture like that:


|Layer (type)              |   Output Shape         |     Param | model.py lines|
|:------------------------:|:---------------------: |:----------|:---------:|
|lambda_1 (Lambda)         |   (None, 160, 320, 3)  |    0      |75|
|cropping2d_1 (Cropping2D) |   (None, 65, 320, 3)   |    0      |77|
|conv2d_1 (Conv2D)         |  (None, 31, 158, 24)   |    1824   |79|
|conv2d_2 (Conv2D)         |  (None, 14, 77, 36)    |    21636  |81|
|conv2d_3 (Conv2D)         |  (None, 5, 37, 48)     |    43248  |83|
|conv2d_4 (Conv2D)         |  (None, 3, 35, 64)     |    27712  |85|
|conv2d_5 (Conv2D)         |  (None, 1, 33, 64)     |    36928  |87|
|flatten_1 (Flatten)       |  (None, 2112)          |    0      |89|
|dropout_1 (Dropout)       |  (None, 2112)          |    0      |91|
|dense_1 (Dense)           |  (None, 100)           |    211300 |93|
|dense_2 (Dense)           |  (None, 50)            |    5050   |95|
|dense_3 (Dense)           |  (None, 10)            |    510    |97|
|dense_4 (Dense)           |  (None, 1)             |    11     |99|


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 91). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 105 shuffle). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I modified the steering angles slightly on the left and right camera angles which brough more data and improved underfitting.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to predict the steering angles based on the driving condition of the vehicle.

My first step was to use a convolution neural network model similar to the naivid model. I thought this model might be appropriate because It was a reliable, mature model. But it still needed to be tuned in a specific scene.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was over fitting. 

To combat the overfitting, I modified the model(adding dropout layer) so that it worked well.

Then I retrained the model, and saved it.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track,such as bridge. To improve the driving behavior in these cases, I converted the image type from BGR to RGB.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first tried to recorde one laps on track one using center lane driving. But it was difficult for me to create a high quality data set. So I used sample data to train the model. Here is an example image of center lane driving:

![alt text][image2]

To augment the data set, I also flipped images and angles thinking that this would increase the amount of data. For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]

After the collection process, I had 38572 number of data points. I then preprocessed this data by function Lambda which was very convenient to make data normalized.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by validation loss and training loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.



```python

```

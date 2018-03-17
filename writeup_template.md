# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


## Collected training dataset and train the model on your side
### The final [training data set](https://drive.google.com/file/d/1bKxHp4ovKr31pqZWSwFzDzCG68Gp4PgB/view?usp=sharing) that I collected from the simulator. It's data_2.tar stored in google drive, please download it and extract the tar ball. 
```sh
tar xvf data_2.tar
```
###If you want to train by the [model.py](https://github.com/erickaoshoulin/CarND-LaneLines-P3/blob/master/model.py). Please specify correct path for csv_file = 'YOUR_EXTRACT_DATA_2_DIR/driving_log.csv' (line 69). 



---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

##### model.py : containing the script to create and train the model. In fact, I trained the model by clone.ipynb. Model.py is saved from clone.ipynb after the model can drive over 1 lap.

##### drive.py : for driving the car in autonomous mode

##### model.h5 : contain a trained convolution neural network 

##### writeup_report.md : Summarize the results


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network which is based on NVIDIA architecture recommended in Udacity Course (model.py lines 300-314) 

The model includes RELU layers to introduce nonlinearity (code line 303, 305, 309 and 310). 
The input data is normalized in the model using a Keras lambda layer (code line 301). Then the data is croprred by using Keras Cropping2D to crop meaningless image for driving (line302)

#### 2. Attempts to reduce overfitting in the model

I did not add dropout layers to reduce overfitting in the model.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 320). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 316).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the course. I thought this model might be appropriate because it's used for real self-driving car by Nvidia.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found Nvidia achitecture is nice with both low training and validation erros.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. For example: the track with opening to muddy road, the track with different left and right scene or the track with high curvature. To improve the driving behavior in these cases, I focus on generating training data set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 300-314) consisted of a convolution neural network.
For more detai, please check on the code.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

I also try to augment the data set by flipping the images and angels. However, it's not enough to train a model to drive 1 lap even with low training and validation error rate.

Then I try to add more data to train the network by driving reversely to prevent overfitting for some kind of sence. Then it's still failed.

Then I repeated this process on track two in order to get more data points.

My memory is out of usage after getting so many data set, then I realize why the course try to teach us use generator to get the training data instead of loading them into memory.

Finally, I find out using multiple cameras is crucial to train the model. I use the simple techniques provided in the course by adding positive and negative correction term for left and right camera seprately (line 112~115). 3 times data points are generated by using this method.
Total 6 times data are agumented by using adding correction term method and image flipping method. 

After the collection process, I had around 40K data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

Then the model can drive autonomously!

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the training error would is not reduced so much when number of epochs are increase. I used an adam optimizer so that manually training the learning rate wasn't necessary.

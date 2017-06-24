#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia.png "NVIDIA Model"
[image2]: ./examples/udacity_steering_cumsum.png "Default data set"
[image3]: ./examples/learning.png "Learning curve of NVIDIA network"
[image4]: ./examples/dataset_steering.png "Recorded data set"
[image5]: ./examples/center.png "Center camera"
[image6]: ./examples/shearing.png "Shearing vs. Rotation"
[image7]: ./examples/dataset_statistics.png "Curvature calculation"
[image8]: ./examples/steering_distance.png "Steering distance calculation"
[image9]: ./examples/steering_distance2.png "Steering distance influence"
[image10]: ./examples/steering_distance3.png "Curvature to steering distance"
[image11]: ./examples/dataset_histogram.png "Steering histogram"
[image12]: ./examples/dataset_histogram2.png "Steering histogram after filter"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* _model.py_ Contains the script to create and train the model
* _preprocessing.py_ Contains methods used to preprocess the data
* _visualization.py_ Contains methods used for visualization of data
* _drive.py_ Contains instructions for driving the car in autonomous mode
* _model.h5_ Contains a trained convolution neural network 
* _writeup_report.md_ Summarizes the results (This file)

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model I have chosen is basically the NVIDIA model from the project introduction.
The NVIDIA base model consists of four convolution layers and four fully connected layers and looks like this:
![NVIDIA Model][image1]

I have made a few adaptions to it:
* The data is normalized outside of the model, because the filtering, pre-processing and augmentation is too complex 
to perform it inside a lambda layer. 
* I squished the input images into 64x64 pixel format in order to reach a bearable training performance of CPU
* Since the flattened layer lost considerable in size due to lower image width, I used SAME padding in last 
  convolution and scaled the first fully connected layer down accordingly.
* I've used Exponential Linear Units ([ELU](https://arxiv.org/abs/1511.07289v1)) as these seem to be the most evolved 
  non-linearities. Like PreLUs and Leaky ReLUs, they are supposed to circumvent the vanishing gradient problem for
  small weights but in addition ELUs preserve most of the filtering capabilies of the ReLU, 
  which we could observe in model visualization of last project. 


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting, but only between the convolutional layers.
I have experienced bad steering behaviour when introducing dropout layers inside of the classifier as well.
This might be due to the fact that the result of our neural network are not probabilties which shall be
maximized but an exact desired value. 

The model was trained and validated on different augmentations of the data set to ensure that the model was not overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Initially I was using the data provided in the
project resources, but it looked not optimal:
* It contained lots of redundant data making training slow (9 laps driven on course 1)
* 5 lap were driven in reverse direction but only 4 in normal direction
* There were parts on the straight sections, where there steering wiggled noticeably

![Default data set][image2]

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was no very spectacular. I started out with the NVIDIA approach
and since it worked perfectly well out of the box I did not change the running system much. 

I had the impression that the network was too large for the simple task at hand, but reducing it's parameters 
and the number of layers had always lead to an increase of final mean scare error. 

I also found with the raw NVIDIA model, that after a certain learn period, the mean squared error curve of the 
training and the validation set would cross, leaving the error on validation growing while the error on test set 
kept improving. This implied that the model was overfitting. 

But after adding dropouts to the convolutions, the learning curve showed a mean squared error of under 0.025 on both 
sets after just 5 training epochs with a batch size of 512:

![Learning curve of NVIDIA network][image3]

The final step was to run the simulator to see how well the car was driving around track one. 
In fact, the vehicle fell of the track whole of the time. This was expected to some degree, as even very small 
steering errors would add up eventually and once the car stood in an unlearnt angle toward the track, 
the network would just keep making unfortunate decisions. 

To improve the driving behavior in these cases, I improved the data set by
* Adding left and right camera images with adapted angles
* Adding sheared versions of camera images with adapted angles
* Adding flipped camera images with adapted angles
 
 At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer 				| Description			 						| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 64x64x3 RGB image   							| 
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 60x60x24 	|
| ELu					|												|
| Max pooling			| 2x2 stride, outputs 30x30x24 				    |
| Dropout				| 0.8 rate 										|
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 26x26x36	|
| ELu					|												|
| Max pooling			| 2x2 stride, outputs 13x13x36					|
| Dropout				| 0.8 rate 										|
| Convolution 5x5		| 1x1 stride, VALID padding, outputs 9x9x48		|
| ELu					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x48					|
| Dropout				| 0.8 rate 										|
| Convolution 3x3		| 1x1 stride, VALID padding, outputs 3x3x64		|
| ELu					|												|
| Convolution 3x3		| 1x1 stride, SAME padding, outputs 3x3x64		|
| ELu					|												|
| Flatten   			| outputs 3x3x64 = 576		        			|
| Fully connected		| outputs 576      						    	|
| ELu					|												|
| Fully connected		| outputs 100      								|
| ELu					|												|
| Fully connected		| outputs 50  									|
| ELu					|												|
| Fully connected		| outputs 10  									|
| ELu					|												|
| Logits				| outputs 1     								|


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded no more than three laps on track one driving forward, the three laps on the
same track driving in reverse direction. I've used the simulator in the lowest resolution on full screen mode with mouse
control to get maximal smooth data:

![Recorded data set][image4]

All of the images recorded have been pushed through the pre-processing pipeline with the main goal to reduce and normalize
the information. Here is an example image of center lane driving with it's preprocessed equivalent:

![Center camera][image5]

I tried to record the vehicle recovering from the sides for a while, but gave up on  it because of the downsides:
* It is a massive amount of work
* It requires perfect record timing or else it will rather decrease data set quality
* It only produces local samples, which might not always generalize so well

So instead, I tried to augment the existing clean data set by applying rotation and shearing to the existing images.
During the experiments, I found out that shearing works especially well. In order to avoid artifacts disturbing the
model, I've repeated the edge pixels during the operation:

![Shearing vs. Rotation][image6]

To further augment the data sat, I also used the images from the left and right camera as well so that every shapshot 
from the simulator would spawn 7 samples:
1) left image, randomly sheared right 15° to 25°
2) left image
3) center image, randomly sheared left 15° to 25°
4) center image
5) center image, randomly sheared right 15° to 25°
6) right image
7) right image, randomly sheared left 15° to 25°

The outer sheared images have been assigned a corrective steering adjustment of 0.5 x the shearing angle, while the inner
(center image) sheared images have been assigned an adjustment of 0.25 x the shearing angle.

The most tricky part was to figure out a steering adjustments for the left and right images, because
* If the correction is too small, the car leaves the track in the curves
* If the correction is too height, the car bounces from one side to the other on straight sections

As a solution, I have implemented a variable steering adjustment. As first step, I was determining the curvature of
the road. In order to do that, I've calculated the mean value of steering over the followup 1.5 seconds 
(model.py: LOOK_AHEAD_TIME) for every steering entry. Herby, I've extracted the time point for each simulator sample
from the image names. The data looks like following (curvature = red):

![Curvature calculation][image7]

Now, I've calculated the angle correction of left and right camera based on trigonometry, which requires a "distance"
on the road ahead to make the angles comparable:

![Steering distance calculation][image8]

I've assume the vehicle to be about one third of the road width, so the cameras distance is one sixth of the 
road width. As reasonable steering distances, I've assumed 1 to 6 road widths ahead. Now using the steering distance,
the amount of angle correction can be controlled smoothly:

![Steering distance influence][image9]

Now all that's left is to connect the curvature of the road to the steering distance for the left and right cams. I did this using a mirrored sigmoid function:

![Curvature to steering distance][image10]

As result, the model should learn to correct by larger amounts, if deviating in front/inside of a curve and to 
correcting by smaller amounts when deviating on a straight sequence.

Furthermore I've flipped all of the so far augmented images thus doubling the size of the data set.
Now only one problem was left, which is that the data is biased toward very small driving angles, which 
causes the model to react to slow with larger steering adjustments in curves.

![Steering histogram][image11]

The histogram shows the amount of steering samples in a one degree resolution (= 1/25).  We can clearly see that samples
in the range -2° to 2° are overrepresented. In order to decrease training time and achieve a better equalized set, I've
implemented a generalized filter formula which achieves a distribution where no history bin shall have more than 2^0.5x 
entries than the mean count of all filled bins.
As a result, the data set distribution looks like following after filtering:

![Steering histogram after filter][image12]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting. 

I've also implemented a generator to provide the data samples into the model in order to 
 decrease memory consumption and as a self exercise.
#  Behavioral Cloning


---

**Behavioral Cloning Project 3**

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

---
### Rubric Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
##### Files Submitted & Code Quality

1. My project includes the following files:

   * model.py containing the script to create and train the model
   * drive.py for driving the car in autonomous mode
   * model.h5 containing a trained convolution neural network
   * writeup_report.md summarizing the results

2. Submission includes functional code

 Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

3. Submission code

 The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

##### Model Architecture

1. I chose the NVIDIA architecture with some small variations as my training model.

 My model consists of 5 convolutional layers followed by 5 fully-connected layers, 3 with dropout.  The final layer is a Dense layer with one output and no activation in order to get a proper estimation of the steering angle, both positive and negative.

 The model includes RELU for most activation functions to introduce nonlinearity.

 See end of section for diagram of architecture.

2. Attempts to reduce overfitting in the model

 The model contains dropout layers after some of the fully-connected layers in order to reduce overfitting.

 The data was split into training (80%) and validation sets (20%) in order to reduce overfitting.

3. Model parameter tuning

 The model used an adam optimizer, so the learning rate was not tuned manually.


| Layer         		|     Description	        					| Output |
|:---------------------:|:---------------------------------------------:|:--:|
| Input         		| 64x64x3 RGB image||
| Layer 1: Convolution 5x5| 24 output, 2x2 stride, SAME padding, RELU activation 	| 32x32x24|
| Layer 2: Convolution 5x5|	36 output, 2x2 stride, SAME padding, RELU activation| 16x16x36|
| Layer 3: Convolution 5x5| 48 output, 2x2 stride, SAME padding, RELU activation 	| 8x8x48|
| Layer 4: Convolution 3x3|	64 output, 2x2 stride, SAME padding, RELU activation| 4x4x64|
| Layer 5: Convolution 3x3| 64 output, 2x2 stride, SAME padding, RELU activation 	| 2x2x64|
| Layer 6: Flatten         			|flatten to 256x1| |
| Layer 7: Dense     	|  	| 256|
| Layer 8: Dropout			|80% keep_prob	| |
| Layer 9: Dense     	|  	|100|
| Layer 10: Dropout			|80% keep_prob	| |
| Layer 11: Dense     	|  	| 50|
| Layer 12: Dropout			|80% keep_prob	| |
| Layer 13: Dense     	| 	|10|
| Layer 14: Dense			|	| 1|

##### Training Data and Preprocessing
1. Augmentation

  In addition to the track data provided by Udacity, I recorded 4 or 5 more loops of data.  Two or three were normal driving around track one, with at least one circuit of track two as well.  I also recorded a loop going the opposite way around track one, in order to generalize the model and not be over-biased to left turns.  Also, I recorded at least one circuit of recovery driving, where I started with the car on or near the side of the ride and steered toward the middle.  After an initial tryout with an earlier model, I also recorded some problem spots on the track, ie, curves where the car had run off the road earlier.  On track two, I tried to drive only in the right lane, which may have helped the model drive on that track up to a point.  After all data collection, I duplicated all data by reading in the images, flipping them and negating the steering angle.  I had about 40,000 data points after all collection.
2. Preprocessing and Generator

 The data required quite a bit of preprocessing.  Initially with all data and processing, the data was too big for TensorFlow, and I determined a generator would be necessary as per the video lessons.  
 I plotted a histogram of the steering data, and determined there was a large bias toward a steering angle of 0.  I randomly dropped 80% of the zero steering angle images to correct for that.  I also randomly chose the left, right or center images in the generator, again in an attempt to provide a more robust dataset for the model.

 The images themselves required specific processing as well, and I followed the steps below to prepare them for the model.    

  *1. Convert to RGB*

 The function imshow reads files into images in a BGR color scheme, so the first step was to correct for that by changing them to RGB.

  *2. Crop*

  The next step was to crop the image, to remove unnecessary data from the top (mostly sky and trees) and bottom (hood of the car).  I removed the top 65 and bottom 25 rows of the image, to leave a 70x320x3 image.

  *3. Resize*

  Since convolutions tend to work better on square images, I resized all the images to 64x64.

  *4. Normalize*

  Finally, I normalized the RGB data so that it would behave better in the model.

  I considered converting the images to grayscale, but the model appeared to perform fine without it.

##### Modifications to drive.py
Since I had done some preprocessing of the data outside of the model in the model file, I needed to mirror this processing in the drive.py file so the model would be seeing the same type of data.  This consisted of nearly the same steps, though the images did not need to be converted to RGB since they were loaded in that format.

I also changed the speed of the simulator to see if the model could handle higher speeds.

#### Conclusion

An initial run of the model without dropout and with fewer data images tended to drive into the water.  I added dropout and also obtained more data, especially recovery situations where the car drove from the side back to the middle of the road again.  I ran the autonomous simulation at 9, 20 and 30 mph on track 1, and the car successfully completed the circuit in all instances.  I also attempted to run track 2 at various speeds.  The model tended to fail fairly quickly at the higher speeds, but nearly made it around at 10mph, failing at a particularly sharp curve at the end of the track.  With more training, I'm confident it could run track 2 as well.

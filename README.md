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

[image1]: ./assets/left_example.jpg "Cropped Left Camera Image"
[image2]: ./assets/right_example.jpg "Cropped Right Camera Image"
[image3]: ./assets/loss.png "Loss after each Epoch"


---
## Files Submitted

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* project_writeup.md

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

## Model Architecture and Training Strategy

### **Model Architecture**

This project employs the end-to-end deep neural network proposed by nVidia. Architecture consists of five convolutional layers and another three fully connected layers. The only real change made to the original model  is that the RGB colour space was used as opposed to the Y’UV colour space. This is done because the driving simulator output is in RGB, yet I believe that the Y’UV colour space would have been more appropriate.  It has the advantage of having one channel that is essentially a grayscale image while the other two channels define the colour of the image. Intuitively this feels like it would make it easier for the network to interpret the images.

A more detailed description of the model architecture can be found in the table below:


|Layer (type)                | Output Shape             | Param #   
|:--------------------------:|:-------------------------:|:-------:|
lambda_1 (Lambda)            |(None, 160, 320, 3)       |0       
cropping2d_1 (Cropping2D)    |(None, 50, 320, 3)        |0         
conv2d_1 (Conv2D)            |(None, 23, 158, 24)       |1824      
conv2d_2 (Conv2D)            |(None, 10, 77, 48)        |28848    
conv2d_3 (Conv2D)            |(None, 8, 75, 64)         |27712    
conv2d_4 (Conv2D)            |(None, 6, 73, 64)         |36928  
flatten_1 (Flatten)          |(None, 28032)             |0       
dropout_1 (Dropout)          |(None, 28032)             |0        
dense_1 (Dense)              |(None, 100)               |2803300   
dropout_2 (Dropout)          |(None, 100)               |0        
dense_2 (Dense)              |(None, 50)                |5050      
dropout_3 (Dropout)          |(None, 50)                |0         
dense_3 (Dense)              |(None, 10)                |510       
dense_4 (Dense)              |(None, 1)                 |11      

_______________________________________________________
- Total params: 2,904,183
- Trainable params: 2,904,183
- Non-trainable params: 0
_______________________________________________________

The Keras lambda and cropping layers are used to preprocess the images prior to training.  The lambda layer normalises and centers the pixel values around 0, while the cropping layer reduces the image size by 70 pixels on top and 40 pixels at the bottom. Cropping by this much leaves only the road visible and reduces the amount of data that needs to be processed by the model:

!['Example of cropped image'][image1]

The convolutional layers include RELU activations to introduce nonlinearity.

To reduce overfitting, a 20% dropout rate was added to the fully connected layers during training.

### **Training**

The model was trained and validated on different data sets to ensure that the model was not overfitting and finally it was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer with a learning rate decay of 1e-6.

### **Training data**

The training data for this project comprises the original dataset provided by Udacity and an additional 2 forward laps, as well as a reverse lap of “Track 1”.  In addition to this, two further laps of Track 2 were recorded for training. The output of the simulator includes a centre, left and right image with the corresponding steering angle at that point in time.

To make the most of the data it was decided to use the left and right images respectively as opposed to the centre image. Since the left and right images are off-centre a correction angle was applied to the measured steering angle. The appropriate magnitude for the correction was determined to be approximately 2.2 degrees by trial and error . The data was then augmented by flipping the images and the recorded steering angles. Unfortunately it was found that using the centre image in addition to these off-center images produced and reliable results so the centre images were not used during training.  Nevertheless, using this technique allowed us to quadruple the number of data points for training our model to 64 136.

Using the left and right offset images with a correction angle for the steering has the advantage that it is not necessary to record recovery maneuvers. All of the data essentially represents a recovery condition.

### **Solution Design Approach**

It was felt that the above mentioned in nVidia network was more than adequate for this application given that it was proven for a similar application in a real world scenario, driving a physical autonomous car. 

This meant that the focus was placed on producing useful training data and ensuring good convergence for the model. 

Nevertheless, after adding training data for "Track 2", the overall loss of the model on the validation data increased twofold. This still highlights a weakness of this approach, where the model needs to be trained for each scenario to work effectively (winter days, desert highways, forest roads). If many different scenarios need to be covered with the same network, it is not expected that this current architecture will be sufficiently deep to work for all of them and it will likely need to become much larger.

Ultimately, this proved to be a successful approach though, with the model navigating “Track 1” and “Track 2” flawlessly after training for 5 epochs. In fact, the exact number of epochs before overfitting begins (when the loss on the training data continues to improve, while the loss on the validation data no longer does) tends to vary for this model during training. The model is therefore saved after each epoch unless the validation loss has not improved.  This ensures that the optimal training result has been reached without needing to monitor it directly.

Below is an example of the trend for the loss on the validation and training sets during training:

!['Progression of loss after each epoch'][image3]

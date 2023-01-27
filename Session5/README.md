--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Requirement**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

5th Assignment is:

You are making 3 versions of your 4th assignment's best model (or pick one from best assignments):

Network with Group Normalization

Network with Layer Normalization

Network with L1 + BN

You MUST:

Write a single model.py file that includes GN/LN/BN and takes an argument to decide which normalization to include

Write a single notebook file to run all the 3 models above for 20 epochs each

Create these graphs:

Graph 1: Test/Validation Loss for all 3 models together

Graph 2: Test/Validation Accuracy for 3 models together

graphs must have proper annotation

Find 10 misclassified images for each of the 3 models, and show them as a 5x2 image matrix in 3 separately annotated images. 

write an explanatory README file that explains:

what is your code all about,

how to perform the 3 normalizations techniques that we covered(cannot use values from the excel sheet shared)

your findings for normalization techniques,

add all your graphs

your 3 collection-of-misclassified-images 

Upload your complete assignment on GitHub and share the link on LMS



--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Normalization - What it is and why is it required**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

We have already used normalization while using any standard neural network libraries (for example, pytorch) while passing the input image to the first layer of the network. We do so by substracting the mean of the data from each pixel value (intensity value) and then dividing it by standard deviation of the data. We do normalization to keep the pixel values within a specific range. The range could be different for different type of data, or for different type of use cases, it can be 0 to 1, -1 to +1, -2 to +2 and anything of such sort.

Above is a general definition. And then why do we need the normalization while passing the data across layers in neural network ?

We know that, in case using neural network, we do not specify the features where we want to put stress on, rather network identifies those during training. From the computer vision topology, if we pass an image for classification (whether specified object is present in the image) or localization (where in the image, the object is), network learns which pixels (pixel intensities) in the image would be decisive in nature, and which are the ones, which has no (very minimal) effect on decision making.

How it does that ? In first forward pass, it calculates the error by comparing the model output with the ground truth. Eventually it back-propagates the error, that means it tries to change the weights (and biases if opted) for each of the layers in order to minimize the error. If we do not put normalization while the data is passing across layers, then eventually some of the intermmediate outputs could be given extra weightage due to having higher value as activation result. Thus training would not be uniform and largely towards the bigger intermmediate outputs. There is a second reason although. At any given point of time during training, we calculate the contribution of each weight in the current error, we say, in the form of d(E)/d(w), where E is the error calculated, w is a specific weight, and we are taking the partial derivative (in order to make a small step towards minimum error). Now if there are intermmediate layers (of weights, and intermmediate outputs) between the said weight and the output, the contribution would be the multiple of all intermmediate partial derivatives in-between. Now, we can understand, any bigger value, when multipled again and again, would be even bigger. Either it can overflow or even unnecessarily overutilize the system memory resources ((we say, exploding gradient issue) and with added bias. We normalize so that training is uniform.

When we say, we subtract mean, we are shifting the data (data distribution) to be centered around zero. And by diving by the standard deviation, we are scaling the data within a specific min-max range.

Now, the next question comes. We said, we are subtracting by the mean of the data and then dividing by the standard deviation. What is the data (population) we are referring and does it vary ? Here lies the concept of various different types of normalizations, batch normalization (BN), layer normalization (LN), group normalization (GN), to name a few. By definition, BN layer transforms each input in the current mini-batch by subtracting the input mean in the current mini-batch and dividing it by the standard deviation. 

We have seen some additional learnable parameters in case of BN. Let's think about those. The following we call as mini-batch mean - 

![image](https://user-images.githubusercontent.com/46663815/215022154-622d2173-b359-4a1b-875a-69ab0c37e028.png)

and the following we call as mini batch variance - 

![image](https://user-images.githubusercontent.com/46663815/215022429-a5277b2c-6799-4df3-9019-3e8712fc3750.png)

and for normalization, we use the following formula - 

![image](https://user-images.githubusercontent.com/46663815/215022545-d93d15ae-56bb-4907-8d60-7910dc12ba43.png)

(please note, square root of variance is standard deviation, and the extra term is added in denominator in order to avoid zero division error.

Here is the final equation - 

![image](https://user-images.githubusercontent.com/46663815/215023052-423bc15b-dec6-40ad-a570-7582b3bd0f4d.png)

This gamma and beta are learnable parameters in case for BN.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Model**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Please refer - 
(as per assignment guidelines, I have created a separate model file and took reference of it (imported) in the notebook.)

https://github.com/partha-goswami/EVA8/blob/main/Session5/model.py

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Notebooks**

Notebook for conducting experiments - 

https://github.com/partha-goswami/EVA8/blob/main/Session5/MNIST_Normalization_Experiments.ipynb

--------------------------------------------------------------------------------------------------------------------------------------------------------------------



--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Experiment Approach**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

We tried with the following normalization approaches - 

1. Batch Normalization with L1 factor as 0, and L2 factor as 0.
2. Batch Normalization with L1 factor as 0, and L2 factor as 0.001.
3. Batch Normalization with L1 factor as 0, and L2 factor as 0.002.
4. Batch Normalization with L1 factor as 0.001, and L2 factor as 0.
5. Batch Normalization with L1 factor as 0.001, and L2 factor as 0.001.
6. Batch Normalization with L1 factor as 0.001, and L2 factor as 0.002.
7. Batch Normalization with L1 factor as 0.002, and L2 factor as 0.
8. Batch Normalization with L1 factor as 0.002, and L2 factor as 0.001.
9. Batch Normalization with L1 factor as 0.002, and L2 factor as 0.002.

10. Layer Normalization with L1 factor as 0, and L2 factor as 0.
11. Layer Normalization with L1 factor as 0, and L2 factor as 0.001.
12. Layer Normalization with L1 factor as 0, and L2 factor as 0.002.
13. Group Normalization with L1 factor as 0, and L2 factor as 0.
14. Group Normalization with L1 factor as 0, and L2 factor as 0.001.
15. Group Normalization with L1 factor as 0, and L2 factor as 0.002.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Experiment Results**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/46663815/215028170-7a55cb93-3549-48e4-8857-c16b46d98e42.png)

![image](https://user-images.githubusercontent.com/46663815/215028262-3942f245-0c50-41ff-85fc-4ea41c1bb138.png)



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

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Notebook for conducting various normalization experiments - 

https://github.com/partha-goswami/EVA8/blob/main/Session5/Session5_Normalization_Experiments.ipynb





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

**Batch Normalization Comparison Results (on test accuracy) - **

![image](https://user-images.githubusercontent.com/46663815/215124127-6bce57fd-9c3f-477f-b41d-89bc2a5851ef.png)

**Layer Normalization Comparison Results (on test accuracy) - **

![image](https://user-images.githubusercontent.com/46663815/215124388-14eb2ff7-ebe9-45f0-bed5-b53ab8f06fb0.png)

**Group Normalization Comparison Results (on test accuracy) - **

![image](https://user-images.githubusercontent.com/46663815/215124587-f46192de-e1c4-4fd7-9d06-4d90a2f0aebe.png)

_We used the legend in the following form - <norm_type>_<L1_factor>_<L2_factor>_Test_Accuracy. So, that means GN_0_0_Test_Accuracy, for example would indicate group normalization, with L1 factor as 0 and L2 factor as 0._

**Combined Normalization Comparison Results (on test accuracy) - **

![image](https://user-images.githubusercontent.com/46663815/215125461-a7639fd9-15f8-48a8-b020-092c2ae1b5db.png)

Here we present a comparison study upon three best models using different normalization types.



--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Misprediction Results**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Misprediction by best Batch Normalization Model - **

The best batch normalization model we are referring is the one that uses L1 factor as 0, and L2 factor as 0.001.

![image](https://user-images.githubusercontent.com/46663815/215126371-d949e88d-972e-4e96-a789-37821d5e92b6.png)


**Misprediction by best Layer Normalization Model - **

The best layer normalization model we are referring is the one that uses L1 factor as 0, and L2 factor as 0.

![image](https://user-images.githubusercontent.com/46663815/215126960-797e49a3-a2c6-466e-af7e-af9e80c64b3f.png)


**Misprediction by best Group Normalization Model - **

The best group normalization model we are referring is the one that uses L1 factor as 0, and L2 factor as 0.

![image](https://user-images.githubusercontent.com/46663815/215127245-3bc478fc-c230-4408-993e-4630e49fa5ec.png)

_Please note, for each misclassified digit, the actual label and the predicted label are shown above the digit._


--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Training Results**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
We are presenting training logs for the three best models using three different normalization types. For training log for other models, please refer https://github.com/partha-goswami/EVA8/blob/main/Session5/Session5_Normalization_Experiments.ipynb

**Training Log for batch normalization model using L1 factor as 0, and L2 factor as 0.001 - ** 

Epoch 1:
  0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.8/dist-packages/torch/utils/data/dataloader.py:554: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(_create_warning_msg(
Loss=0.49202385544776917 Batch_id=468 Accuracy=63.45: 100%|██████████| 469/469 [00:34<00:00, 13.48it/s]
Test set: Average loss: 0.2637, Accuracy: 9499/10000 (94.99%)

Epoch 2:
Loss=0.27910783886909485 Batch_id=468 Accuracy=91.79: 100%|██████████| 469/469 [00:34<00:00, 13.46it/s]
Test set: Average loss: 0.0662, Accuracy: 9845/10000 (98.45%)

Epoch 3:
Loss=0.2833850383758545 Batch_id=468 Accuracy=95.03: 100%|██████████| 469/469 [00:34<00:00, 13.45it/s]
Test set: Average loss: 0.0418, Accuracy: 9886/10000 (98.86%)

Epoch 4:
Loss=0.12308276444673538 Batch_id=468 Accuracy=96.22: 100%|██████████| 469/469 [00:35<00:00, 13.27it/s]
Test set: Average loss: 0.0408, Accuracy: 9868/10000 (98.68%)

Epoch 5:
Loss=0.06830643862485886 Batch_id=468 Accuracy=96.81: 100%|██████████| 469/469 [00:34<00:00, 13.43it/s]
Test set: Average loss: 0.0285, Accuracy: 9919/10000 (99.19%)

Epoch 6:
Loss=0.16469836235046387 Batch_id=468 Accuracy=97.10: 100%|██████████| 469/469 [00:36<00:00, 12.92it/s]
Test set: Average loss: 0.0358, Accuracy: 9882/10000 (98.82%)

Epoch 7:
Loss=0.06487025320529938 Batch_id=468 Accuracy=97.29: 100%|██████████| 469/469 [00:35<00:00, 13.11it/s]
Test set: Average loss: 0.0259, Accuracy: 9922/10000 (99.22%)

Epoch 8:
Loss=0.06728126108646393 Batch_id=468 Accuracy=97.46: 100%|██████████| 469/469 [00:35<00:00, 13.32it/s]
Test set: Average loss: 0.0287, Accuracy: 9906/10000 (99.06%)

Epoch 9:
Loss=0.04668973386287689 Batch_id=468 Accuracy=97.52: 100%|██████████| 469/469 [00:35<00:00, 13.19it/s]
Test set: Average loss: 0.0255, Accuracy: 9929/10000 (99.29%)

Epoch 10:
Loss=0.02658090554177761 Batch_id=468 Accuracy=97.59: 100%|██████████| 469/469 [00:35<00:00, 13.25it/s]
Test set: Average loss: 0.0251, Accuracy: 9923/10000 (99.23%)

Epoch 11:
Loss=0.04944537207484245 Batch_id=468 Accuracy=97.80: 100%|██████████| 469/469 [00:35<00:00, 13.32it/s]
Test set: Average loss: 0.0262, Accuracy: 9924/10000 (99.24%)

Epoch 12:
Loss=0.08205502480268478 Batch_id=468 Accuracy=97.78: 100%|██████████| 469/469 [00:35<00:00, 13.27it/s]
Test set: Average loss: 0.0260, Accuracy: 9927/10000 (99.27%)

Epoch 13:
Loss=0.0927739068865776 Batch_id=468 Accuracy=97.93: 100%|██████████| 469/469 [00:36<00:00, 12.79it/s]
Test set: Average loss: 0.0238, Accuracy: 9930/10000 (99.30%)

Epoch 14:
Loss=0.14265035092830658 Batch_id=468 Accuracy=98.00: 100%|██████████| 469/469 [00:35<00:00, 13.21it/s]
Test set: Average loss: 0.0218, Accuracy: 9932/10000 (99.32%)

Epoch 15:
Loss=0.03880902752280235 Batch_id=468 Accuracy=98.05: 100%|██████████| 469/469 [00:35<00:00, 13.25it/s]
Test set: Average loss: 0.0200, Accuracy: 9939/10000 (99.39%)

Epoch 16:
Loss=0.035227399319410324 Batch_id=468 Accuracy=98.19: 100%|██████████| 469/469 [00:35<00:00, 13.16it/s]
Test set: Average loss: 0.0183, Accuracy: 9938/10000 (99.38%)

Epoch 17:
Loss=0.030972430482506752 Batch_id=468 Accuracy=98.29: 100%|██████████| 469/469 [00:35<00:00, 13.13it/s]
Test set: Average loss: 0.0182, Accuracy: 9941/10000 (99.41%)

Epoch 18:
Loss=0.04370884224772453 Batch_id=468 Accuracy=98.47: 100%|██████████| 469/469 [00:35<00:00, 13.27it/s]
Test set: Average loss: 0.0164, Accuracy: 9951/10000 (99.51%)

Epoch 19:
Loss=0.011540145613253117 Batch_id=468 Accuracy=98.45: 100%|██████████| 469/469 [00:35<00:00, 13.27it/s]
Test set: Average loss: 0.0165, Accuracy: 9951/10000 (99.51%)

Epoch 20:
Loss=0.028170542791485786 Batch_id=468 Accuracy=98.45: 100%|██████████| 469/469 [00:37<00:00, 12.65it/s]
Test set: Average loss: 0.0163, Accuracy: 9954/10000 (99.54%)


**Training Log for layer normalization model using L1 factor as 0, and L2 factor as 0**

Epoch 1:
Loss=0.17670778930187225 Batch_id=937 Accuracy=72.72: 100%|██████████| 938/938 [01:16<00:00, 12.25it/s]
Test set: Average loss: 0.1559, Accuracy: 9642/10000 (96.42%)

Epoch 2:
Loss=0.18419621884822845 Batch_id=937 Accuracy=94.03: 100%|██████████| 938/938 [01:17<00:00, 12.16it/s]
Test set: Average loss: 0.0643, Accuracy: 9812/10000 (98.12%)

Epoch 3:
Loss=0.18432538211345673 Batch_id=937 Accuracy=95.85: 100%|██████████| 938/938 [01:17<00:00, 12.17it/s]
Test set: Average loss: 0.0420, Accuracy: 9873/10000 (98.73%)

Epoch 4:
Loss=0.18362677097320557 Batch_id=937 Accuracy=96.44: 100%|██████████| 938/938 [01:16<00:00, 12.24it/s]
Test set: Average loss: 0.0382, Accuracy: 9893/10000 (98.93%)

Epoch 5:
Loss=0.09606991708278656 Batch_id=937 Accuracy=96.89: 100%|██████████| 938/938 [01:16<00:00, 12.27it/s]
Test set: Average loss: 0.0433, Accuracy: 9892/10000 (98.92%)

Epoch 6:
Loss=0.013062437996268272 Batch_id=937 Accuracy=97.29: 100%|██████████| 938/938 [01:17<00:00, 12.18it/s]
Test set: Average loss: 0.0405, Accuracy: 9889/10000 (98.89%)

Epoch 7:
Loss=0.06257122755050659 Batch_id=937 Accuracy=97.63: 100%|██████████| 938/938 [01:15<00:00, 12.40it/s]
Test set: Average loss: 0.0346, Accuracy: 9914/10000 (99.14%)

Epoch 8:
Loss=0.07140842080116272 Batch_id=937 Accuracy=97.67: 100%|██████████| 938/938 [01:15<00:00, 12.38it/s]
Test set: Average loss: 0.0332, Accuracy: 9904/10000 (99.04%)

Epoch 9:
Loss=0.011693554930388927 Batch_id=937 Accuracy=97.81: 100%|██████████| 938/938 [01:17<00:00, 12.16it/s]
Test set: Average loss: 0.0264, Accuracy: 9908/10000 (99.08%)

Epoch 10:
Loss=0.020994270220398903 Batch_id=937 Accuracy=97.91: 100%|██████████| 938/938 [01:16<00:00, 12.20it/s]
Test set: Average loss: 0.0291, Accuracy: 9912/10000 (99.12%)

Epoch 11:
Loss=0.007920579053461552 Batch_id=937 Accuracy=98.03: 100%|██████████| 938/938 [01:16<00:00, 12.21it/s]
Test set: Average loss: 0.0245, Accuracy: 9926/10000 (99.26%)

Epoch 12:
Loss=0.00791252963244915 Batch_id=937 Accuracy=98.17: 100%|██████████| 938/938 [01:17<00:00, 12.17it/s]
Test set: Average loss: 0.0271, Accuracy: 9915/10000 (99.15%)

Epoch 13:
Loss=0.10385610163211823 Batch_id=937 Accuracy=98.27: 100%|██████████| 938/938 [01:16<00:00, 12.25it/s]
Test set: Average loss: 0.0228, Accuracy: 9922/10000 (99.22%)

Epoch 14:
Loss=0.05467239022254944 Batch_id=937 Accuracy=98.32: 100%|██████████| 938/938 [01:16<00:00, 12.28it/s]
Test set: Average loss: 0.0214, Accuracy: 9931/10000 (99.31%)

Epoch 15:
Loss=0.030536891892552376 Batch_id=937 Accuracy=98.49: 100%|██████████| 938/938 [01:15<00:00, 12.39it/s]
Test set: Average loss: 0.0217, Accuracy: 9938/10000 (99.38%)

Epoch 16:
Loss=0.13934683799743652 Batch_id=937 Accuracy=98.49: 100%|██████████| 938/938 [01:16<00:00, 12.29it/s]
Test set: Average loss: 0.0201, Accuracy: 9940/10000 (99.40%)

Epoch 17:
Loss=0.025102540850639343 Batch_id=937 Accuracy=98.63: 100%|██████████| 938/938 [01:17<00:00, 12.13it/s]
Test set: Average loss: 0.0220, Accuracy: 9930/10000 (99.30%)

Epoch 18:
Loss=0.025957344099879265 Batch_id=937 Accuracy=98.62: 100%|██████████| 938/938 [01:16<00:00, 12.19it/s]
Test set: Average loss: 0.0206, Accuracy: 9935/10000 (99.35%)

Epoch 19:
Loss=0.002684277016669512 Batch_id=937 Accuracy=98.75: 100%|██████████| 938/938 [01:15<00:00, 12.39it/s]
Test set: Average loss: 0.0185, Accuracy: 9935/10000 (99.35%)

Epoch 20:
Loss=0.011199524626135826 Batch_id=937 Accuracy=98.72: 100%|██████████| 938/938 [01:17<00:00, 12.16it/s]
Test set: Average loss: 0.0188, Accuracy: 9934/10000 (99.34%)


**Training Log for group normalization model using L1 factor as 0, and L2 factor as 0**

Epoch 1:
Loss=0.4919760227203369 Batch_id=468 Accuracy=61.16: 100%|██████████| 469/469 [00:37<00:00, 12.62it/s]
Test set: Average loss: 0.2825, Accuracy: 9465/10000 (94.65%)

Epoch 2:
Loss=0.3325135409832001 Batch_id=468 Accuracy=91.53: 100%|██████████| 469/469 [00:37<00:00, 12.38it/s]
Test set: Average loss: 0.0841, Accuracy: 9795/10000 (97.95%)

Epoch 3:
Loss=0.15378402173519135 Batch_id=468 Accuracy=94.58: 100%|██████████| 469/469 [00:36<00:00, 12.77it/s]
Test set: Average loss: 0.0603, Accuracy: 9837/10000 (98.37%)

Epoch 4:
Loss=0.08383096009492874 Batch_id=468 Accuracy=95.81: 100%|██████████| 469/469 [00:36<00:00, 12.76it/s]
Test set: Average loss: 0.0452, Accuracy: 9875/10000 (98.75%)

Epoch 5:
Loss=0.13007916510105133 Batch_id=468 Accuracy=96.45: 100%|██████████| 469/469 [00:36<00:00, 12.73it/s]
Test set: Average loss: 0.0396, Accuracy: 9885/10000 (98.85%)

Epoch 6:
Loss=0.13766932487487793 Batch_id=468 Accuracy=96.90: 100%|██████████| 469/469 [00:36<00:00, 12.78it/s]
Test set: Average loss: 0.0439, Accuracy: 9871/10000 (98.71%)

Epoch 7:
Loss=0.14465047419071198 Batch_id=468 Accuracy=97.14: 100%|██████████| 469/469 [00:37<00:00, 12.49it/s]
Test set: Average loss: 0.0512, Accuracy: 9851/10000 (98.51%)

Epoch 8:
Loss=0.12460916489362717 Batch_id=468 Accuracy=97.25: 100%|██████████| 469/469 [00:36<00:00, 12.77it/s]
Test set: Average loss: 0.0316, Accuracy: 9902/10000 (99.02%)

Epoch 9:
Loss=0.06588899344205856 Batch_id=468 Accuracy=97.66: 100%|██████████| 469/469 [00:36<00:00, 12.74it/s]
Test set: Average loss: 0.0276, Accuracy: 9910/10000 (99.10%)

Epoch 10:
Loss=0.06914158910512924 Batch_id=468 Accuracy=97.62: 100%|██████████| 469/469 [00:36<00:00, 12.79it/s]
Test set: Average loss: 0.0323, Accuracy: 9902/10000 (99.02%)

Epoch 11:
Loss=0.013776391744613647 Batch_id=468 Accuracy=97.73: 100%|██████████| 469/469 [00:38<00:00, 12.19it/s]
Test set: Average loss: 0.0320, Accuracy: 9909/10000 (99.09%)

Epoch 12:
Loss=0.05352669954299927 Batch_id=468 Accuracy=97.88: 100%|██████████| 469/469 [00:36<00:00, 12.70it/s]
Test set: Average loss: 0.0236, Accuracy: 9928/10000 (99.28%)

Epoch 13:
Loss=0.03978570178151131 Batch_id=468 Accuracy=97.90: 100%|██████████| 469/469 [00:36<00:00, 12.74it/s]
Test set: Average loss: 0.0283, Accuracy: 9916/10000 (99.16%)

Epoch 14:
Loss=0.035416558384895325 Batch_id=468 Accuracy=97.99: 100%|██████████| 469/469 [00:36<00:00, 12.68it/s]
Test set: Average loss: 0.0201, Accuracy: 9939/10000 (99.39%)

Epoch 15:
Loss=0.06372503936290741 Batch_id=468 Accuracy=98.07: 100%|██████████| 469/469 [00:38<00:00, 12.09it/s]
Test set: Average loss: 0.0241, Accuracy: 9927/10000 (99.27%)

Epoch 16:
Loss=0.055238593369722366 Batch_id=468 Accuracy=98.17: 100%|██████████| 469/469 [00:37<00:00, 12.61it/s]
Test set: Average loss: 0.0198, Accuracy: 9936/10000 (99.36%)

Epoch 17:
Loss=0.1440947949886322 Batch_id=468 Accuracy=98.28: 100%|██████████| 469/469 [00:36<00:00, 12.76it/s]
Test set: Average loss: 0.0191, Accuracy: 9950/10000 (99.50%)

Epoch 18:
Loss=0.06714516133069992 Batch_id=468 Accuracy=98.42: 100%|██████████| 469/469 [00:36<00:00, 12.82it/s]
Test set: Average loss: 0.0181, Accuracy: 9950/10000 (99.50%)

Epoch 19:
Loss=0.024716945365071297 Batch_id=468 Accuracy=98.44: 100%|██████████| 469/469 [00:38<00:00, 12.30it/s]
Test set: Average loss: 0.0173, Accuracy: 9953/10000 (99.53%)

Epoch 20:
Loss=0.04863129183650017 Batch_id=468 Accuracy=98.50: 100%|██████████| 469/469 [00:36<00:00, 12.79it/s]
Test set: Average loss: 0.0178, Accuracy: 9952/10000 (99.52%)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Synopsis of Assignment**

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/46663815/212736577-8ee393df-6644-4204-b4ed-20e4891c49d5.png)


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
**First Submission - Various Calculations and Analysis**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Target** - 

Test Accuracy - 99.4

Number of Parameters to be used - 8000

**Results** - 

Best Test Accuracy Got - 99.26

Number of Parameters Used - 6.37 M

**Analysis** - 

<img width="817" alt="Analysis-CalculatingRF" src="https://user-images.githubusercontent.com/46663815/212836291-1fcaf07b-08f3-4e96-a493-b87959e1c364.png">


<img width="641" alt="Analysis-TrainingAndTest" src="https://user-images.githubusercontent.com/46663815/212836365-338a1f7b-5bc3-4ed3-bb62-79a230ff29be.png">


**Training log** - 

EPOCH: 0
Loss=0.05034107342362404 Batch_id=468 Accuracy=88.08: 100%|██████████| 469/469 [00:17<00:00, 26.13it/s]
Test set: Average loss: 0.0636, Accuracy: 9798/10000 (97.98%)

EPOCH: 1
Loss=0.02247297577559948 Batch_id=468 Accuracy=98.36: 100%|██████████| 469/469 [00:17<00:00, 26.82it/s]
Test set: Average loss: 0.0350, Accuracy: 9874/10000 (98.74%)

EPOCH: 2
Loss=0.04779127240180969 Batch_id=468 Accuracy=98.92: 100%|██████████| 469/469 [00:17<00:00, 26.49it/s]
Test set: Average loss: 0.0324, Accuracy: 9890/10000 (98.90%)

EPOCH: 3
Loss=0.0049747563898563385 Batch_id=468 Accuracy=99.21: 100%|██████████| 469/469 [00:17<00:00, 26.38it/s]
Test set: Average loss: 0.0287, Accuracy: 9904/10000 (99.04%)

EPOCH: 4
Loss=0.014100850559771061 Batch_id=468 Accuracy=99.46: 100%|██████████| 469/469 [00:18<00:00, 25.53it/s]
Test set: Average loss: 0.0279, Accuracy: 9898/10000 (98.98%)

EPOCH: 5
Loss=0.005061745177954435 Batch_id=468 Accuracy=99.50: 100%|██████████| 469/469 [00:17<00:00, 26.49it/s]
Test set: Average loss: 0.0308, Accuracy: 9901/10000 (99.01%)

EPOCH: 6
Loss=0.038377318531274796 Batch_id=468 Accuracy=99.69: 100%|██████████| 469/469 [00:18<00:00, 25.40it/s]
Test set: Average loss: 0.0277, Accuracy: 9915/10000 (99.15%)

EPOCH: 7
Loss=0.005827886518090963 Batch_id=468 Accuracy=99.74: 100%|██████████| 469/469 [00:17<00:00, 26.49it/s]
Test set: Average loss: 0.0278, Accuracy: 9917/10000 (99.17%)

EPOCH: 8
Loss=0.0024357286747545004 Batch_id=468 Accuracy=99.77: 100%|██████████| 469/469 [00:17<00:00, 26.24it/s]
Test set: Average loss: 0.0274, Accuracy: 9924/10000 (99.24%)

EPOCH: 9
Loss=0.001400904729962349 Batch_id=468 Accuracy=99.84: 100%|██████████| 469/469 [00:17<00:00, 26.46it/s]
Test set: Average loss: 0.0291, Accuracy: 9926/10000 (99.26%)

EPOCH: 10
Loss=0.005444033071398735 Batch_id=468 Accuracy=99.84: 100%|██████████| 469/469 [00:17<00:00, 26.24it/s]
Test set: Average loss: 0.0298, Accuracy: 9923/10000 (99.23%)

EPOCH: 11
Loss=0.0007956892368383706 Batch_id=468 Accuracy=99.83: 100%|██████████| 469/469 [00:17<00:00, 26.27it/s]
Test set: Average loss: 0.0289, Accuracy: 9923/10000 (99.23%)

EPOCH: 12
Loss=1.8141072359867394e-05 Batch_id=468 Accuracy=99.91: 100%|██████████| 469/469 [00:17<00:00, 26.35it/s]
Test set: Average loss: 0.0317, Accuracy: 9923/10000 (99.23%)

EPOCH: 13
Loss=0.0017670737579464912 Batch_id=468 Accuracy=99.96: 100%|██████████| 469/469 [00:17<00:00, 26.47it/s]
Test set: Average loss: 0.0336, Accuracy: 9918/10000 (99.18%)

EPOCH: 14
Loss=3.553391798050143e-05 Batch_id=468 Accuracy=99.92: 100%|██████████| 469/469 [00:17<00:00, 26.20it/s]
Test set: Average loss: 0.0320, Accuracy: 9919/10000 (99.19%)



**Final Comments** - 

1. After 6th, the scope of further accuracy improvement via training is less than 0.5 %, so there's little to no chance of further test accuracy improvement.

2. After 4th epoch, training accuracy is lot higher and almost near to 100 % and the difference between train and test accuracy is quite high. This gives an idea that may be the model is overfitting, or we have made more complex model for a simpler problem.

3. We have not reached test accuracy, parameters used are huge in numbers. Plus we can see a sign of over fitting. Looks like, with this model design, we won't be able to cross 99.4 target accuracy even if we train for few more epochs as further actual training options seem limited.



--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Second Submission - Various Calculations and Analysis**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Target** - 

Test Accuracy - 99.4

Number of Parameters to be used - 8000

**Results** - 

Best Test Accuracy Got - 98.24

Number of Parameters Used - 7530

**Analysis** - 

<img width="843" alt="RFCalculation" src="https://user-images.githubusercontent.com/46663815/212920597-3029be78-cff9-445b-98f0-d10034173559.png">


<img width="822" alt="TrainVsTestAccuracy" src="https://user-images.githubusercontent.com/46663815/212921068-3f79f4c7-4a53-4bdf-8ce2-533275428f8c.png">


**Training Log** - 

EPOCH: 0
Loss=2.2949304580688477 Batch_id=468 Accuracy=10.97: 100%|██████████| 469/469 [00:16<00:00, 28.29it/s]
Test set: Average loss: 2.3009, Accuracy: 1135/10000 (11.35%)

EPOCH: 1
Loss=2.2900540828704834 Batch_id=468 Accuracy=11.24: 100%|██████████| 469/469 [00:15<00:00, 30.28it/s]
Test set: Average loss: 2.2992, Accuracy: 1135/10000 (11.35%)

EPOCH: 2
Loss=1.88473641872406 Batch_id=468 Accuracy=17.65: 100%|██████████| 469/469 [00:14<00:00, 32.96it/s]
Test set: Average loss: 1.9821, Accuracy: 2486/10000 (24.86%)

EPOCH: 3
Loss=0.9276673197746277 Batch_id=468 Accuracy=47.84: 100%|██████████| 469/469 [00:14<00:00, 32.59it/s]
Test set: Average loss: 0.7377, Accuracy: 7504/10000 (75.04%)

EPOCH: 4
Loss=0.3634425103664398 Batch_id=468 Accuracy=80.71: 100%|██████████| 469/469 [00:14<00:00, 32.90it/s]
Test set: Average loss: 0.3860, Accuracy: 8780/10000 (87.80%)

EPOCH: 5
Loss=0.25479447841644287 Batch_id=468 Accuracy=90.87: 100%|██████████| 469/469 [00:14<00:00, 32.58it/s]
Test set: Average loss: 0.1794, Accuracy: 9448/10000 (94.48%)

EPOCH: 6
Loss=0.06549306958913803 Batch_id=468 Accuracy=94.83: 100%|██████████| 469/469 [00:15<00:00, 30.35it/s]
Test set: Average loss: 0.1452, Accuracy: 9538/10000 (95.38%)

EPOCH: 7
Loss=0.1386694461107254 Batch_id=468 Accuracy=95.99: 100%|██████████| 469/469 [00:14<00:00, 32.80it/s]
Test set: Average loss: 0.1100, Accuracy: 9658/10000 (96.58%)

EPOCH: 8
Loss=0.052027519792318344 Batch_id=468 Accuracy=96.81: 100%|██████████| 469/469 [00:14<00:00, 33.40it/s]
Test set: Average loss: 0.1261, Accuracy: 9595/10000 (95.95%)

EPOCH: 9
Loss=0.0959506630897522 Batch_id=468 Accuracy=97.30: 100%|██████████| 469/469 [00:14<00:00, 32.75it/s]
Test set: Average loss: 0.0672, Accuracy: 9786/10000 (97.86%)

EPOCH: 10
Loss=0.04913612827658653 Batch_id=468 Accuracy=97.68: 100%|██████████| 469/469 [00:14<00:00, 32.63it/s]
Test set: Average loss: 0.0744, Accuracy: 9786/10000 (97.86%)

EPOCH: 11
Loss=0.15558584034442902 Batch_id=468 Accuracy=97.91: 100%|██████████| 469/469 [00:14<00:00, 33.17it/s]
Test set: Average loss: 0.0707, Accuracy: 9770/10000 (97.70%)

EPOCH: 12
Loss=0.10839885473251343 Batch_id=468 Accuracy=97.98: 100%|██████████| 469/469 [00:14<00:00, 33.01it/s]
Test set: Average loss: 0.0608, Accuracy: 9813/10000 (98.13%)

EPOCH: 13
Loss=0.03812249377369881 Batch_id=468 Accuracy=98.08: 100%|██████████| 469/469 [00:14<00:00, 32.79it/s]
Test set: Average loss: 0.0599, Accuracy: 9813/10000 (98.13%)

EPOCH: 14
Loss=0.08262475579977036 Batch_id=468 Accuracy=98.28: 100%|██████████| 469/469 [00:15<00:00, 30.27it/s]
Test set: Average loss: 0.0547, Accuracy: 9824/10000 (98.24%)


**Comments** - 

1. If we closely follow the training and test accuracy progression, we would see that the model is learning but apparently slowly. We do not see any sign of overfitting but looks like the model is underfitted.

2. Steady increase of train and test accuracy further states that the skeleton of the model is correct, but we would have to add few more components for faster convergence.

3. At the end of last epoch, we have a training accuracy of 98.28, so the model should still learn and there is enough chance that test accuracy might improve if we take few more epochs.

4. Its good that we could keep the number of parameters below 8000. Currently the number of parameters used are 7530. We assume, this might increase a bit while we would add the batch normalization (for keeping mean, std etc.) in next set of submissions, but we hope, we would be able to keep the number of parameters within 8000.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Third Submission - Various Calculations and Analysis**
**Added Batch Normalization**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Target** - 

Test Accuracy - 99.4

Number of Parameters to be used - 8000

**Results** - 

Best Test Accuracy Got - 99.10

Number of Parameters Used - 7770

**Analysis** - 

For adding batch normalization, RF and other dimensionality calculations remain same like 2nd submission.

<img width="843" alt="RFCalculation" src="https://user-images.githubusercontent.com/46663815/212968666-f4922e49-9e01-4fce-9fb8-129adf1637f2.png">

<img width="616" alt="TrainAccuracy" src="https://user-images.githubusercontent.com/46663815/212968773-4017e626-ff2a-4dcd-a78c-20df5ddf1c1a.png">

<img width="556" alt="TestAccuracy" src="https://user-images.githubusercontent.com/46663815/212968841-57bc6ed4-b84d-49a1-a402-aedb14a98b52.png">


**Training Log** -

EPOCH: 0
Loss=0.08500686287879944 Batch_id=468 Accuracy=91.89: 100%|██████████| 469/469 [00:19<00:00, 23.87it/s]

Test set: Average loss: 0.0995, Accuracy: 9714/10000 (97.14%)

EPOCH: 1
Loss=0.07431977242231369 Batch_id=468 Accuracy=97.82: 100%|██████████| 469/469 [00:14<00:00, 32.84it/s]

Test set: Average loss: 0.0708, Accuracy: 9797/10000 (97.97%)

EPOCH: 2
Loss=0.11868520826101303 Batch_id=468 Accuracy=98.36: 100%|██████████| 469/469 [00:14<00:00, 32.26it/s]

Test set: Average loss: 0.0440, Accuracy: 9866/10000 (98.66%)

EPOCH: 3
Loss=0.11161275953054428 Batch_id=468 Accuracy=98.67: 100%|██████████| 469/469 [00:14<00:00, 32.99it/s]

Test set: Average loss: 0.0359, Accuracy: 9887/10000 (98.87%)

EPOCH: 4
Loss=0.04690796509385109 Batch_id=468 Accuracy=98.80: 100%|██████████| 469/469 [00:14<00:00, 32.46it/s]

Test set: Average loss: 0.0447, Accuracy: 9857/10000 (98.57%)

EPOCH: 5
Loss=0.08476153016090393 Batch_id=468 Accuracy=98.90: 100%|██████████| 469/469 [00:14<00:00, 32.54it/s]

Test set: Average loss: 0.0385, Accuracy: 9877/10000 (98.77%)

EPOCH: 6
Loss=0.010062671266496181 Batch_id=468 Accuracy=98.96: 100%|██████████| 469/469 [00:15<00:00, 30.17it/s]

Test set: Average loss: 0.0340, Accuracy: 9893/10000 (98.93%)

EPOCH: 7
Loss=0.06732156127691269 Batch_id=468 Accuracy=99.00: 100%|██████████| 469/469 [00:14<00:00, 33.13it/s]

Test set: Average loss: 0.0278, Accuracy: 9910/10000 (99.10%)

EPOCH: 8
Loss=0.0017466951394453645 Batch_id=468 Accuracy=99.12: 100%|██████████| 469/469 [00:14<00:00, 33.07it/s]

Test set: Average loss: 0.0327, Accuracy: 9897/10000 (98.97%)

EPOCH: 9
Loss=0.0023736897855997086 Batch_id=468 Accuracy=99.19: 100%|██████████| 469/469 [00:13<00:00, 33.63it/s]

Test set: Average loss: 0.0342, Accuracy: 9883/10000 (98.83%)

EPOCH: 10
Loss=0.008427104912698269 Batch_id=468 Accuracy=99.19: 100%|██████████| 469/469 [00:14<00:00, 33.33it/s]

Test set: Average loss: 0.0336, Accuracy: 9888/10000 (98.88%)

EPOCH: 11
Loss=0.04995420202612877 Batch_id=468 Accuracy=99.27: 100%|██████████| 469/469 [00:13<00:00, 33.55it/s]

Test set: Average loss: 0.0386, Accuracy: 9879/10000 (98.79%)

EPOCH: 12
Loss=0.06207267567515373 Batch_id=468 Accuracy=99.30: 100%|██████████| 469/469 [00:14<00:00, 33.29it/s]

Test set: Average loss: 0.0286, Accuracy: 9908/10000 (99.08%)

EPOCH: 13
Loss=0.02146131545305252 Batch_id=468 Accuracy=99.35: 100%|██████████| 469/469 [00:14<00:00, 32.79it/s]

Test set: Average loss: 0.0295, Accuracy: 9903/10000 (99.03%)

EPOCH: 14
Loss=0.07630743086338043 Batch_id=468 Accuracy=99.41: 100%|██████████| 469/469 [00:13<00:00, 33.54it/s]

Test set: Average loss: 0.0295, Accuracy: 9907/10000 (99.07%)


**Comments** - 

1. Batch normalization has improved both average train and test accuracy and the model is converging faster.

2. Batch normalization increased the number of parameters by around 200. Still we are using only 7700 odd parameters, well within the specified limit.





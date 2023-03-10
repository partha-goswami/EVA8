-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Synopsis of Assignment**

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/46663815/212736577-8ee393df-6644-4204-b4ed-20e4891c49d5.png)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Final Submission Details**

Best Test Accuracy - 99.42

Model Parameters Used - 7640

ipynb file - Session4_SixthSubmission.ipynb

_Next we would be providing details as we progressed in submission, starting from first to sixth (final submission)_

--------------------------------------------------------------------------------------------------------------------------------------------------------------------










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



--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Fourth Submission - Various Calculations and Analysis**
**Added Dropout/regularization**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Target** - 

Test Accuracy - 99.4

Number of Parameters to be used - 8000

**Results** - 

Best Test Accuracy Got - 99.13

Number of Parameters Used - 7770

**Analysis** - 

For adding batch normalization, RF and other dimensionality calculations remain same like 2nd submission.

<img width="843" alt="RFCalculation" src="https://user-images.githubusercontent.com/46663815/213115726-9e0b091c-fa6e-4f12-90bc-36ddd143e10d.png">

<img width="690" alt="TrainAccuracy" src="https://user-images.githubusercontent.com/46663815/213115793-3cb175c5-f83f-4aa5-9143-1812e717a0b0.png">

<img width="698" alt="TestAccuracy" src="https://user-images.githubusercontent.com/46663815/213115841-782ed985-1d26-41bd-ae6d-4cc1bdbca45c.png">

<img width="613" alt="TrainMinusTestAccuracy" src="https://user-images.githubusercontent.com/46663815/213115880-69523eec-9741-410d-b05c-835a60f6c276.png">



**Training Log** -

EPOCH: 0
Loss=0.09455364942550659 Batch_id=468 Accuracy=89.64: 100%|██████████| 469/469 [00:18<00:00, 25.80it/s]

Test set: Average loss: 0.0913, Accuracy: 9718/10000 (97.18%)

EPOCH: 1
Loss=0.1343652456998825 Batch_id=468 Accuracy=96.88: 100%|██████████| 469/469 [00:15<00:00, 30.80it/s]

Test set: Average loss: 0.0663, Accuracy: 9801/10000 (98.01%)

EPOCH: 2
Loss=0.1577949970960617 Batch_id=468 Accuracy=97.74: 100%|██████████| 469/469 [00:15<00:00, 30.40it/s]

Test set: Average loss: 0.0467, Accuracy: 9856/10000 (98.56%)

EPOCH: 3
Loss=0.12714873254299164 Batch_id=468 Accuracy=98.05: 100%|██████████| 469/469 [00:15<00:00, 31.00it/s]

Test set: Average loss: 0.0507, Accuracy: 9849/10000 (98.49%)

EPOCH: 4
Loss=0.07319943606853485 Batch_id=468 Accuracy=98.28: 100%|██████████| 469/469 [00:15<00:00, 31.07it/s]

Test set: Average loss: 0.0557, Accuracy: 9836/10000 (98.36%)

EPOCH: 5
Loss=0.037692248821258545 Batch_id=468 Accuracy=98.37: 100%|██████████| 469/469 [00:15<00:00, 30.04it/s]

Test set: Average loss: 0.0409, Accuracy: 9877/10000 (98.77%)

EPOCH: 6
Loss=0.03277427703142166 Batch_id=468 Accuracy=98.47: 100%|██████████| 469/469 [00:15<00:00, 30.79it/s]

Test set: Average loss: 0.0417, Accuracy: 9868/10000 (98.68%)

EPOCH: 7
Loss=0.1435360461473465 Batch_id=468 Accuracy=98.57: 100%|██████████| 469/469 [00:16<00:00, 29.29it/s]

Test set: Average loss: 0.0408, Accuracy: 9869/10000 (98.69%)

EPOCH: 8
Loss=0.012902501039206982 Batch_id=468 Accuracy=98.64: 100%|██████████| 469/469 [00:15<00:00, 30.80it/s]

Test set: Average loss: 0.0334, Accuracy: 9900/10000 (99.00%)

EPOCH: 9
Loss=0.0072029666043818 Batch_id=468 Accuracy=98.76: 100%|██████████| 469/469 [00:15<00:00, 30.89it/s]

Test set: Average loss: 0.0312, Accuracy: 9901/10000 (99.01%)

EPOCH: 10
Loss=0.0102304145693779 Batch_id=468 Accuracy=98.77: 100%|██████████| 469/469 [00:15<00:00, 30.99it/s]

Test set: Average loss: 0.0388, Accuracy: 9887/10000 (98.87%)

EPOCH: 11
Loss=0.06496897339820862 Batch_id=468 Accuracy=98.82: 100%|██████████| 469/469 [00:15<00:00, 30.64it/s]

Test set: Average loss: 0.0337, Accuracy: 9899/10000 (98.99%)

EPOCH: 12
Loss=0.0763937458395958 Batch_id=468 Accuracy=98.78: 100%|██████████| 469/469 [00:15<00:00, 30.75it/s]

Test set: Average loss: 0.0307, Accuracy: 9913/10000 (99.13%)

EPOCH: 13
Loss=0.031496454030275345 Batch_id=468 Accuracy=98.83: 100%|██████████| 469/469 [00:15<00:00, 30.83it/s]

Test set: Average loss: 0.0426, Accuracy: 9879/10000 (98.79%)

EPOCH: 14
Loss=0.11864728480577469 Batch_id=468 Accuracy=98.93: 100%|██████████| 469/469 [00:15<00:00, 30.42it/s]

Test set: Average loss: 0.0333, Accuracy: 9898/10000 (98.98%)


**Comments** - 

1. Looks like introducing dropoff has regularized the training and correlation between train and test accuracy. Detailed other comparison statistics are described alongside the charts.

2. Although we are within limit in using number of model parameters, still out peak test accuracy hasn't touched the target accuracy. We would work upon that in next series of submissions.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Fifth Submission - Various Calculations and Analysis**
**Added Image Augmentation**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Target** - 

Test Accuracy - 99.4

Number of Parameters to be used - 8000

**Results** - 

Best Test Accuracy Got - 99.15

Number of Parameters Used - 7770


**Analysis** - 

RF Calculation would be same like previous as we have not added any extra layers in the model than our previous submission.

<img width="843" alt="RFCalculation" src="https://user-images.githubusercontent.com/46663815/213200888-9041d8da-3242-4a64-8343-3e24709bbf61.png">

<img width="727" alt="TrainAccuracyComparison" src="https://user-images.githubusercontent.com/46663815/213200990-f0026394-6c0d-47f0-83e8-04c995349edf.png">

<img width="618" alt="TestAccuracyComparison" src="https://user-images.githubusercontent.com/46663815/213201039-6bb14fb3-195c-4695-9f7e-8905639d966b.png">


**Training Log** -


EPOCH: 0
Loss=0.2732337415218353 Batch_id=468 Accuracy=88.16: 100%|██████████| 469/469 [00:17<00:00, 27.40it/s]

Test set: Average loss: 0.0872, Accuracy: 9745/10000 (97.45%)

EPOCH: 1
Loss=0.16329407691955566 Batch_id=468 Accuracy=96.56: 100%|██████████| 469/469 [00:17<00:00, 26.23it/s]

Test set: Average loss: 0.0748, Accuracy: 9776/10000 (97.76%)

EPOCH: 2
Loss=0.04289648309350014 Batch_id=468 Accuracy=97.33: 100%|██████████| 469/469 [00:17<00:00, 27.25it/s]

Test set: Average loss: 0.0474, Accuracy: 9847/10000 (98.47%)

EPOCH: 3
Loss=0.11414932459592819 Batch_id=468 Accuracy=97.66: 100%|██████████| 469/469 [00:16<00:00, 27.85it/s]

Test set: Average loss: 0.0401, Accuracy: 9877/10000 (98.77%)

EPOCH: 4
Loss=0.03618963062763214 Batch_id=468 Accuracy=97.88: 100%|██████████| 469/469 [00:17<00:00, 27.47it/s]

Test set: Average loss: 0.0491, Accuracy: 9848/10000 (98.48%)

EPOCH: 5
Loss=0.07262670248746872 Batch_id=468 Accuracy=98.02: 100%|██████████| 469/469 [00:16<00:00, 27.86it/s]

Test set: Average loss: 0.0388, Accuracy: 9886/10000 (98.86%)

EPOCH: 6
Loss=0.10475189238786697 Batch_id=468 Accuracy=98.09: 100%|██████████| 469/469 [00:17<00:00, 27.21it/s]

Test set: Average loss: 0.0370, Accuracy: 9888/10000 (98.88%)

EPOCH: 7
Loss=0.014746402390301228 Batch_id=468 Accuracy=98.29: 100%|██████████| 469/469 [00:16<00:00, 27.62it/s]

Test set: Average loss: 0.0315, Accuracy: 9900/10000 (99.00%)

EPOCH: 8
Loss=0.07913126051425934 Batch_id=468 Accuracy=98.27: 100%|██████████| 469/469 [00:17<00:00, 27.55it/s]

Test set: Average loss: 0.0305, Accuracy: 9914/10000 (99.14%)

EPOCH: 9
Loss=0.015233825892210007 Batch_id=468 Accuracy=98.38: 100%|██████████| 469/469 [00:17<00:00, 27.37it/s]

Test set: Average loss: 0.0322, Accuracy: 9903/10000 (99.03%)

EPOCH: 10
Loss=0.023798728361725807 Batch_id=468 Accuracy=98.47: 100%|██████████| 469/469 [00:17<00:00, 27.53it/s]

Test set: Average loss: 0.0301, Accuracy: 9904/10000 (99.04%)

EPOCH: 11
Loss=0.07839823514223099 Batch_id=468 Accuracy=98.50: 100%|██████████| 469/469 [00:17<00:00, 27.33it/s]

Test set: Average loss: 0.0270, Accuracy: 9910/10000 (99.10%)

EPOCH: 12
Loss=0.16510041058063507 Batch_id=468 Accuracy=98.49: 100%|██████████| 469/469 [00:18<00:00, 25.76it/s]

Test set: Average loss: 0.0317, Accuracy: 9907/10000 (99.07%)

EPOCH: 13
Loss=0.014580834656953812 Batch_id=468 Accuracy=98.56: 100%|██████████| 469/469 [00:16<00:00, 27.67it/s]

Test set: Average loss: 0.0274, Accuracy: 9915/10000 (99.15%)

EPOCH: 14
Loss=0.012958255596458912 Batch_id=468 Accuracy=98.58: 100%|██████████| 469/469 [00:17<00:00, 27.45it/s]

Test set: Average loss: 0.0276, Accuracy: 9915/10000 (99.15%)


**Comments** - 

1. One obvious model performance improvement we can see. The test accuracy hasn't ever come down below 99 percent after epoch number 7. This indicates, the inference accuracy is not by chance, rather the model consistently improved.

2. Still we have not touched target test accuracy, so we would do further optimizations and would observe the result in next submissions.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Final Submission - Various Calculations and Analysis**

--------------------------------------------------------------------------------------------------------------------------------------------------------------------


**Target** - 

Test Accuracy - 99.4

Number of Parameters to be used - 8000

**Results** - 

Best Test Accuracy Got - 99.42

Number of Parameters Used - 7640


**Analysis** - 

RF Calculation would be same like previous as we have not added any extra layers in the model than our previous submission.

<img width="843" alt="RFCalculation" src="https://user-images.githubusercontent.com/46663815/213706538-087c2fde-698d-4baf-bcef-2eb39cb09c5b.png">

<img width="772" alt="TrainVsTest" src="https://user-images.githubusercontent.com/46663815/213706592-4c4a7bb4-759f-459f-be19-fda4d5ef52b1.png">


**Training Log** -

EPOCH: 0
Loss=0.27262410521507263 Batch_id=468 Accuracy=81.43: 100%|██████████| 469/469 [00:38<00:00, 12.11it/s]
Test set: Average loss: 0.0810, Accuracy: 9754/10000 (97.54%)

EPOCH: 1
Loss=0.25788623094558716 Batch_id=468 Accuracy=94.41: 100%|██████████| 469/469 [00:35<00:00, 13.04it/s]
Test set: Average loss: 0.0645, Accuracy: 9785/10000 (97.85%)

EPOCH: 2
Loss=0.13935916125774384 Batch_id=468 Accuracy=95.76: 100%|██████████| 469/469 [00:35<00:00, 13.17it/s]
Test set: Average loss: 0.0649, Accuracy: 9798/10000 (97.98%)

EPOCH: 3
Loss=0.11832254379987717 Batch_id=468 Accuracy=96.35: 100%|██████████| 469/469 [00:34<00:00, 13.65it/s]
Test set: Average loss: 0.0569, Accuracy: 9823/10000 (98.23%)

EPOCH: 4
Loss=0.12303054332733154 Batch_id=468 Accuracy=96.71: 100%|██████████| 469/469 [00:34<00:00, 13.62it/s]
Test set: Average loss: 0.0346, Accuracy: 9883/10000 (98.83%)

EPOCH: 5
Loss=0.05285732075572014 Batch_id=468 Accuracy=97.04: 100%|██████████| 469/469 [00:34<00:00, 13.64it/s]
Test set: Average loss: 0.0428, Accuracy: 9867/10000 (98.67%)

EPOCH: 6
Loss=0.0554448626935482 Batch_id=468 Accuracy=97.20: 100%|██████████| 469/469 [00:35<00:00, 13.09it/s]
Test set: Average loss: 0.0272, Accuracy: 9914/10000 (99.14%)

EPOCH: 7
Loss=0.04172109439969063 Batch_id=468 Accuracy=97.32: 100%|██████████| 469/469 [00:34<00:00, 13.51it/s]
Test set: Average loss: 0.0309, Accuracy: 9914/10000 (99.14%)

EPOCH: 8
Loss=0.06748926639556885 Batch_id=468 Accuracy=97.53: 100%|██████████| 469/469 [00:34<00:00, 13.74it/s]
Test set: Average loss: 0.0246, Accuracy: 9922/10000 (99.22%)

EPOCH: 9
Loss=0.12525376677513123 Batch_id=468 Accuracy=97.55: 100%|██████████| 469/469 [00:34<00:00, 13.64it/s]
Test set: Average loss: 0.0247, Accuracy: 9929/10000 (99.29%)

EPOCH: 10
Loss=0.1687525361776352 Batch_id=468 Accuracy=97.60: 100%|██████████| 469/469 [00:34<00:00, 13.68it/s]
Test set: Average loss: 0.0244, Accuracy: 9929/10000 (99.29%)

EPOCH: 11
Loss=0.09009213000535965 Batch_id=468 Accuracy=97.62: 100%|██████████| 469/469 [00:36<00:00, 13.00it/s]
Test set: Average loss: 0.0230, Accuracy: 9926/10000 (99.26%)

EPOCH: 12
Loss=0.12872128188610077 Batch_id=468 Accuracy=97.81: 100%|██████████| 469/469 [00:34<00:00, 13.56it/s]
Test set: Average loss: 0.0213, Accuracy: 9938/10000 (99.38%)

EPOCH: 13
Loss=0.035853102803230286 Batch_id=468 Accuracy=97.75: 100%|██████████| 469/469 [00:34<00:00, 13.71it/s]
Test set: Average loss: 0.0179, Accuracy: 9941/10000 (99.41%)

EPOCH: 14
Loss=0.08171842247247696 Batch_id=468 Accuracy=97.84: 100%|██████████| 469/469 [00:34<00:00, 13.71it/s]
Test set: Average loss: 0.0168, Accuracy: 9942/10000 (99.42%)


**Comments** - 

We could achieve target test accuracy while using lesser than 8k parameters. Moreover in the convoluions we used, we have made bias as false, hence there's less number of trainable parameters. We used SGD optimizer with StepLR for reaching the desired accuracy. SGD's learning rate is kept as 0.01, and momentum we kept at 0.9. Step size we used as 6, with gamma value as 0.1.

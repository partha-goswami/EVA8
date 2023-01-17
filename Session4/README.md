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


**Comments** - 

1. If we closely follow the training and test accuracy progression, we would see that the model is learning but apparently slowly. We do not see any sign of overfitting but looks like the model is underfitted.

2. Steady increase of train and test accuracy further states that the skeleton of the model is correct, but we would have to add few more components for faster convergence.

3. At the end of last epoch, we have a training accuracy of 98.28, so the model should still learn and there is enough chance that test accuracy might improve if we take few more epochs.

4. Its good that we could keep the number of parameters below 8000. Currently the number of parameters used are 7530. We assume, this might increase a bit while we would add the batch normalization (for keeping mean, std etc.) in next set of submissions, but we hope, we would be able to keep the number of parameters within 8000.

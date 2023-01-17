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







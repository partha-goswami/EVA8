**Problem Statement**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/46663815/212443988-d8443196-6dae-462f-b94e-5af8c5a4f1d4.png)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**What is the purpose of doing this exercise from a layman's angle**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Everywhere we hear about neural networks and in neural network, we have heard that it can automatically select features and provide importance on them. So that we manually don't require to calculate like correlation metrics etc. In this excel, we showed how neural network does so for one data, for fixed number of epochs (epoch means number of times we have seen the total data). Once we have one pass output and true label/value for output, it can calculate the contribution of various features (weights, biases in our case, for ease or simplicity, we have counted bias as zero) on the error. That means, it can identify the positive, negative and no correlations (no correlation means negligible weights or zero weights). The more data we supply, the weights are better approximated to get the best results under specified constraints.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Neural Network Design and initial values Referred**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/46663815/212444035-08d9acc6-dc90-403b-81c7-3a7a572e9cbc.png)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Calculations**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/46663815/212444096-7d182ccf-a2e3-4886-9977-d6299f166d58.png)
![image](https://user-images.githubusercontent.com/46663815/212444120-9849abb9-0245-4b4a-9778-dccb4e5f721b.png)
![image](https://user-images.githubusercontent.com/46663815/212444141-925a10fc-f303-4297-a57d-3e402607a888.png)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Epoch vs total error chart when learning rate (eta) is 0.1**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/46663815/212444997-064b2074-e397-4b98-b97a-32762a3b24dd.png)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Epoch vs total error chart when learning rate (eta) is 0.2**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/46663815/212445200-2682851a-4aed-4c50-9d04-314c3d601249.png)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Epoch vs total error chart when learning rate (eta) is 0.5**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/46663815/212445258-4b8a9309-bbed-49bd-9c14-1620ba14f115.png)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Epoch vs total error chart when learning rate (eta) is 0.8**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/46663815/212445336-507d3bb9-9d0c-4771-96e1-b4c551b9efef.png)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Epoch vs total error chart when learning rate (eta) is 1**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/46663815/212445423-ca6e65a5-3abf-4961-b95b-4b68da843dff.png)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Epoch vs total error chart when learning rate (eta) is 2**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/46663815/212445479-237f994a-e49d-4514-9bf7-4457dd23ad2d.png)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Steps for calculation**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

We have two inputs (i1 and i2). And we have taken initial values for weights starting from w1 till w8. We have calculated initial error and then propagated it back. We used learning rates as 0.1, 0.2, 0.5, 0.8, 1 and 2. We calculated rate of change of error with nearest output weight first and then used multiplication of partial derivatives to get the rate of change of error with respect to further weights from output. Please refer calculations section for details. If we plot the features (here weights, assuming bias is zero) and the total error, it can be mathematically proven that, at any point in that graph, the partial derivative would mean the tangent drawn at that point towards the minimum error. We would move towards that according to our learning rate (eta).

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Limitations**
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

For illustration purpose, we showed by taking one data and calculating errors and backpropagating etc. at the excel. Usually the count of data is much more and neural network is more stable when fed with enormous data. In those cases, we take data in a batch, calculate error and back-propagate.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

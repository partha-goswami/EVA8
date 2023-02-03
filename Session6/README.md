&#x1F537;**Problem Definition**&#x1F537;

While using Depthwise separable convolution and dilated convolution blocks, we would have a design a neural network that achieves at least 85 % accuracy on CIFAR10 dataset. Total number of model parameters should be lesser than 200k, moreover we should be using GAP as one of the layers. The receptive field should be above 44. We would have to use a code that has the following basic blocks as convolution block1, convolution block2, convolution block3, convolution block 4, and output.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------



&#x1F537;**Model**&#x1F537;


https://github.com/partha-goswami/EVA8/blob/main/Session6/models/model.py

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**Notebook File**&#x1F537;


https://github.com/partha-goswami/EVA8/blob/main/Session6/Session6_Advanced_Topics_Assignment_Solution.ipynb

---------------------------------------------------------------------------------------------------------------------------------------------------------------------



&#x1F537;**Model Parameters and Final RF**&#x1F537;


**Model Parameters**: 97284

**Final Receptive Fiield at the last layer**: 109

**Best Validation Accuracy Obtained**: 85.20 (72 epochs)

_(Input, output and RF calculations are done at each step and is detailed at model.py file above. I have used both dilution and depthwise separable convolution.)_

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**Training Log**&#x1F537;

Epoch 1:
Loss=1.9371486902236938 Batch_id=195 Accuracy=29.61: 100%|██████████| 196/196 [00:16<00:00, 12.12it/s]

Test set: Average loss: 0.0068, Accuracy: 4361/10000 (43.61%)

Epoch 2:
Loss=1.7106716632843018 Batch_id=195 Accuracy=41.54: 100%|██████████| 196/196 [00:16<00:00, 12.05it/s]

Test set: Average loss: 0.0060, Accuracy: 5139/10000 (51.39%)

Epoch 3:
Loss=1.5164453983306885 Batch_id=195 Accuracy=48.18: 100%|██████████| 196/196 [00:16<00:00, 11.85it/s]

Test set: Average loss: 0.0053, Accuracy: 5709/10000 (57.09%)

Epoch 4:
Loss=1.2748647928237915 Batch_id=195 Accuracy=53.26: 100%|██████████| 196/196 [00:16<00:00, 12.01it/s]

Test set: Average loss: 0.0049, Accuracy: 6098/10000 (60.98%)

Epoch 5:
Loss=1.2323343753814697 Batch_id=195 Accuracy=56.88: 100%|██████████| 196/196 [00:16<00:00, 12.21it/s]

Test set: Average loss: 0.0043, Accuracy: 6571/10000 (65.71%)

Epoch 6:
Loss=1.280673623085022 Batch_id=195 Accuracy=59.32: 100%|██████████| 196/196 [00:16<00:00, 12.00it/s]

Test set: Average loss: 0.0043, Accuracy: 6530/10000 (65.30%)

Epoch 7:
Loss=1.2698378562927246 Batch_id=195 Accuracy=61.85: 100%|██████████| 196/196 [00:16<00:00, 12.11it/s]

Test set: Average loss: 0.0038, Accuracy: 6942/10000 (69.42%)

Epoch 8:
Loss=1.017128825187683 Batch_id=195 Accuracy=63.38: 100%|██████████| 196/196 [00:16<00:00, 12.20it/s]

Test set: Average loss: 0.0036, Accuracy: 7053/10000 (70.53%)

Epoch 9:
Loss=1.0090224742889404 Batch_id=195 Accuracy=64.97: 100%|██████████| 196/196 [00:15<00:00, 12.28it/s]

Test set: Average loss: 0.0036, Accuracy: 7035/10000 (70.35%)

Epoch 10:
Loss=1.0494353771209717 Batch_id=195 Accuracy=65.98: 100%|██████████| 196/196 [00:16<00:00, 12.06it/s]

Test set: Average loss: 0.0036, Accuracy: 7026/10000 (70.26%)

Epoch 11:
Loss=0.9911742210388184 Batch_id=195 Accuracy=66.69: 100%|██████████| 196/196 [00:16<00:00, 11.94it/s]

Test set: Average loss: 0.0033, Accuracy: 7216/10000 (72.16%)

Epoch 12:
Loss=1.1618266105651855 Batch_id=195 Accuracy=67.42: 100%|██████████| 196/196 [00:16<00:00, 12.04it/s]

Test set: Average loss: 0.0032, Accuracy: 7337/10000 (73.37%)

Epoch 13:
Loss=0.9455658793449402 Batch_id=195 Accuracy=67.71: 100%|██████████| 196/196 [00:17<00:00, 11.07it/s]

Test set: Average loss: 0.0033, Accuracy: 7281/10000 (72.81%)

Epoch 14:
Loss=1.0856261253356934 Batch_id=195 Accuracy=68.13: 100%|██████████| 196/196 [00:16<00:00, 12.01it/s]

Test set: Average loss: 0.0030, Accuracy: 7541/10000 (75.41%)

Epoch 15:
Loss=0.9132341146469116 Batch_id=195 Accuracy=68.76: 100%|██████████| 196/196 [00:16<00:00, 12.01it/s]

Test set: Average loss: 0.0033, Accuracy: 7220/10000 (72.20%)

Epoch 16:
Loss=0.805076003074646 Batch_id=195 Accuracy=69.42: 100%|██████████| 196/196 [00:16<00:00, 11.96it/s]

Test set: Average loss: 0.0028, Accuracy: 7569/10000 (75.69%)

Epoch 17:
Loss=0.7440102100372314 Batch_id=195 Accuracy=69.56: 100%|██████████| 196/196 [00:16<00:00, 12.01it/s]

Test set: Average loss: 0.0030, Accuracy: 7503/10000 (75.03%)

Epoch 18:
Loss=0.9029043316841125 Batch_id=195 Accuracy=69.53: 100%|██████████| 196/196 [00:16<00:00, 11.67it/s]

Test set: Average loss: 0.0030, Accuracy: 7465/10000 (74.65%)

Epoch 19:
Loss=0.7610498666763306 Batch_id=195 Accuracy=69.84: 100%|██████████| 196/196 [00:16<00:00, 11.67it/s]

Test set: Average loss: 0.0030, Accuracy: 7493/10000 (74.93%)

Epoch 20:
Loss=0.7394888997077942 Batch_id=195 Accuracy=70.00: 100%|██████████| 196/196 [00:17<00:00, 11.53it/s]

Test set: Average loss: 0.0028, Accuracy: 7668/10000 (76.68%)

Epoch 21:
Loss=1.1128368377685547 Batch_id=195 Accuracy=70.47: 100%|██████████| 196/196 [00:16<00:00, 11.66it/s]

Test set: Average loss: 0.0031, Accuracy: 7371/10000 (73.71%)

Epoch 22:
Loss=0.9401184916496277 Batch_id=195 Accuracy=70.49: 100%|██████████| 196/196 [00:16<00:00, 11.72it/s]

Test set: Average loss: 0.0029, Accuracy: 7527/10000 (75.27%)

Epoch 23:
Loss=0.7843496799468994 Batch_id=195 Accuracy=70.72: 100%|██████████| 196/196 [00:16<00:00, 11.59it/s]

Test set: Average loss: 0.0029, Accuracy: 7568/10000 (75.68%)

Epoch 24:
Loss=0.7335059642791748 Batch_id=195 Accuracy=70.81: 100%|██████████| 196/196 [00:16<00:00, 11.79it/s]

Test set: Average loss: 0.0034, Accuracy: 7188/10000 (71.88%)

Epoch 25:
Loss=1.1092424392700195 Batch_id=195 Accuracy=71.02: 100%|██████████| 196/196 [00:17<00:00, 10.91it/s]

Test set: Average loss: 0.0032, Accuracy: 7402/10000 (74.02%)

Epoch 26:
Loss=1.0462005138397217 Batch_id=195 Accuracy=71.05: 100%|██████████| 196/196 [00:16<00:00, 11.73it/s]

Test set: Average loss: 0.0028, Accuracy: 7639/10000 (76.39%)

Epoch 27:
Loss=1.0045673847198486 Batch_id=195 Accuracy=71.14: 100%|██████████| 196/196 [00:16<00:00, 11.78it/s]

Test set: Average loss: 0.0026, Accuracy: 7735/10000 (77.35%)

Epoch 28:
Loss=0.8853341341018677 Batch_id=195 Accuracy=71.27: 100%|██████████| 196/196 [00:16<00:00, 11.98it/s]

Test set: Average loss: 0.0027, Accuracy: 7647/10000 (76.47%)

Epoch 29:
Loss=0.9365090131759644 Batch_id=195 Accuracy=71.17: 100%|██████████| 196/196 [00:16<00:00, 11.78it/s]

Test set: Average loss: 0.0028, Accuracy: 7673/10000 (76.73%)

Epoch 30:
Loss=0.916951060295105 Batch_id=195 Accuracy=71.50: 100%|██████████| 196/196 [00:16<00:00, 11.85it/s]

Test set: Average loss: 0.0030, Accuracy: 7444/10000 (74.44%)

Epoch 31:
Loss=1.0645740032196045 Batch_id=195 Accuracy=71.37: 100%|██████████| 196/196 [00:16<00:00, 11.82it/s]

Test set: Average loss: 0.0028, Accuracy: 7599/10000 (75.99%)

Epoch 32:
Loss=1.024559736251831 Batch_id=195 Accuracy=71.75: 100%|██████████| 196/196 [00:16<00:00, 11.77it/s]

Test set: Average loss: 0.0033, Accuracy: 7246/10000 (72.46%)

Epoch 33:
Loss=0.9699963331222534 Batch_id=195 Accuracy=71.66: 100%|██████████| 196/196 [00:16<00:00, 11.71it/s]

Test set: Average loss: 0.0028, Accuracy: 7629/10000 (76.29%)

Epoch 34:
Loss=0.8779830932617188 Batch_id=195 Accuracy=71.98: 100%|██████████| 196/196 [00:16<00:00, 11.73it/s]

Test set: Average loss: 0.0026, Accuracy: 7665/10000 (76.65%)

Epoch 35:
Loss=0.7946022748947144 Batch_id=195 Accuracy=71.52: 100%|██████████| 196/196 [00:16<00:00, 11.71it/s]

Test set: Average loss: 0.0031, Accuracy: 7438/10000 (74.38%)

Epoch 36:
Loss=0.8246749043464661 Batch_id=195 Accuracy=71.94: 100%|██████████| 196/196 [00:16<00:00, 11.69it/s]

Test set: Average loss: 0.0028, Accuracy: 7632/10000 (76.32%)

Epoch 37:
Loss=0.8907915353775024 Batch_id=195 Accuracy=71.92: 100%|██████████| 196/196 [00:17<00:00, 10.91it/s]

Test set: Average loss: 0.0030, Accuracy: 7517/10000 (75.17%)

Epoch 38:
Loss=0.8354768753051758 Batch_id=195 Accuracy=72.15: 100%|██████████| 196/196 [00:16<00:00, 11.71it/s]

Test set: Average loss: 0.0025, Accuracy: 7892/10000 (78.92%)

Epoch 39:
Loss=0.8708763122558594 Batch_id=195 Accuracy=72.14: 100%|██████████| 196/196 [00:16<00:00, 11.77it/s]

Test set: Average loss: 0.0027, Accuracy: 7754/10000 (77.54%)

Epoch 40:
Loss=0.738598644733429 Batch_id=195 Accuracy=72.46: 100%|██████████| 196/196 [00:16<00:00, 11.85it/s]

Test set: Average loss: 0.0026, Accuracy: 7822/10000 (78.22%)

Epoch 41:
Loss=0.7667247653007507 Batch_id=195 Accuracy=72.83: 100%|██████████| 196/196 [00:16<00:00, 11.77it/s]

Test set: Average loss: 0.0026, Accuracy: 7737/10000 (77.37%)

Epoch 42:
Loss=1.013182282447815 Batch_id=195 Accuracy=72.69: 100%|██████████| 196/196 [00:16<00:00, 11.74it/s]

Test set: Average loss: 0.0027, Accuracy: 7702/10000 (77.02%)

Epoch 43:
Loss=0.92280513048172 Batch_id=195 Accuracy=72.81: 100%|██████████| 196/196 [00:16<00:00, 11.75it/s]

Test set: Average loss: 0.0030, Accuracy: 7523/10000 (75.23%)

Epoch 44:
Loss=0.8970339894294739 Batch_id=195 Accuracy=73.08: 100%|██████████| 196/196 [00:16<00:00, 11.88it/s]

Test set: Average loss: 0.0025, Accuracy: 7903/10000 (79.03%)

Epoch 45:
Loss=0.6552590131759644 Batch_id=195 Accuracy=72.96: 100%|██████████| 196/196 [00:16<00:00, 11.75it/s]

Test set: Average loss: 0.0024, Accuracy: 7978/10000 (79.78%)

Epoch 46:
Loss=0.8169801831245422 Batch_id=195 Accuracy=73.31: 100%|██████████| 196/196 [00:16<00:00, 11.73it/s]

Test set: Average loss: 0.0024, Accuracy: 7930/10000 (79.30%)

Epoch 47:
Loss=0.8236644864082336 Batch_id=195 Accuracy=73.69: 100%|██████████| 196/196 [00:16<00:00, 11.69it/s]

Test set: Average loss: 0.0024, Accuracy: 7954/10000 (79.54%)

Epoch 48:
Loss=0.9900811314582825 Batch_id=195 Accuracy=73.46: 100%|██████████| 196/196 [00:16<00:00, 11.68it/s]

Test set: Average loss: 0.0024, Accuracy: 7984/10000 (79.84%)

Epoch 49:
Loss=0.8291366696357727 Batch_id=195 Accuracy=74.04: 100%|██████████| 196/196 [00:17<00:00, 10.94it/s]

Test set: Average loss: 0.0027, Accuracy: 7737/10000 (77.37%)

Epoch 50:
Loss=0.6869806051254272 Batch_id=195 Accuracy=73.99: 100%|██████████| 196/196 [00:16<00:00, 11.59it/s]

Test set: Average loss: 0.0024, Accuracy: 7911/10000 (79.11%)

Epoch 51:
Loss=0.7001036405563354 Batch_id=195 Accuracy=74.43: 100%|██████████| 196/196 [00:16<00:00, 11.67it/s]

Test set: Average loss: 0.0022, Accuracy: 8118/10000 (81.18%)

Epoch 52:
Loss=0.7368819713592529 Batch_id=195 Accuracy=74.44: 100%|██████████| 196/196 [00:16<00:00, 11.60it/s]

Test set: Average loss: 0.0024, Accuracy: 7932/10000 (79.32%)

Epoch 53:
Loss=0.7317963242530823 Batch_id=195 Accuracy=74.60: 100%|██████████| 196/196 [00:16<00:00, 11.63it/s]

Test set: Average loss: 0.0023, Accuracy: 8031/10000 (80.31%)

Epoch 54:
Loss=0.5556869506835938 Batch_id=195 Accuracy=74.85: 100%|██████████| 196/196 [00:16<00:00, 11.82it/s]

Test set: Average loss: 0.0022, Accuracy: 8098/10000 (80.98%)

Epoch 55:
Loss=1.094988465309143 Batch_id=195 Accuracy=75.23: 100%|██████████| 196/196 [00:16<00:00, 11.76it/s]

Test set: Average loss: 0.0022, Accuracy: 8106/10000 (81.06%)

Epoch 56:
Loss=0.6839620471000671 Batch_id=195 Accuracy=75.13: 100%|██████████| 196/196 [00:16<00:00, 11.83it/s]

Test set: Average loss: 0.0023, Accuracy: 8090/10000 (80.90%)

Epoch 57:
Loss=0.7589752078056335 Batch_id=195 Accuracy=75.68: 100%|██████████| 196/196 [00:16<00:00, 12.03it/s]

Test set: Average loss: 0.0024, Accuracy: 8035/10000 (80.35%)

Epoch 58:
Loss=0.7050309777259827 Batch_id=195 Accuracy=75.82: 100%|██████████| 196/196 [00:16<00:00, 12.03it/s]

Test set: Average loss: 0.0023, Accuracy: 8074/10000 (80.74%)

Epoch 59:
Loss=0.5596480965614319 Batch_id=195 Accuracy=76.24: 100%|██████████| 196/196 [00:16<00:00, 11.86it/s]

Test set: Average loss: 0.0021, Accuracy: 8215/10000 (82.15%)

Epoch 60:
Loss=0.7529387474060059 Batch_id=195 Accuracy=76.55: 100%|██████████| 196/196 [00:16<00:00, 11.85it/s]

Test set: Average loss: 0.0020, Accuracy: 8196/10000 (81.96%)

Epoch 61:
Loss=0.6616877913475037 Batch_id=195 Accuracy=76.52: 100%|██████████| 196/196 [00:17<00:00, 11.09it/s]

Test set: Average loss: 0.0020, Accuracy: 8246/10000 (82.46%)

Epoch 62:
Loss=0.5258587598800659 Batch_id=195 Accuracy=77.19: 100%|██████████| 196/196 [00:16<00:00, 11.72it/s]

Test set: Average loss: 0.0020, Accuracy: 8310/10000 (83.10%)

Epoch 63:
Loss=0.8879419565200806 Batch_id=195 Accuracy=77.40: 100%|██████████| 196/196 [00:16<00:00, 11.68it/s]

Test set: Average loss: 0.0020, Accuracy: 8322/10000 (83.22%)

Epoch 64:
Loss=0.8439589738845825 Batch_id=195 Accuracy=77.55: 100%|██████████| 196/196 [00:16<00:00, 11.68it/s]

Test set: Average loss: 0.0019, Accuracy: 8395/10000 (83.95%)

Epoch 65:
Loss=0.6423452496528625 Batch_id=195 Accuracy=77.89: 100%|██████████| 196/196 [00:16<00:00, 11.74it/s]

Test set: Average loss: 0.0019, Accuracy: 8377/10000 (83.77%)

Epoch 66:
Loss=0.6325924396514893 Batch_id=195 Accuracy=78.18: 100%|██████████| 196/196 [00:16<00:00, 11.93it/s]

Test set: Average loss: 0.0019, Accuracy: 8398/10000 (83.98%)

Epoch 67:
Loss=0.6742538213729858 Batch_id=195 Accuracy=78.79: 100%|██████████| 196/196 [00:16<00:00, 11.85it/s]

Test set: Average loss: 0.0019, Accuracy: 8390/10000 (83.90%)

Epoch 68:
Loss=0.7806583046913147 Batch_id=195 Accuracy=79.01: 100%|██████████| 196/196 [00:16<00:00, 11.59it/s]

Test set: Average loss: 0.0018, Accuracy: 8464/10000 (84.64%)

Epoch 69:
Loss=0.5567256808280945 Batch_id=195 Accuracy=79.12: 100%|██████████| 196/196 [00:16<00:00, 11.74it/s]

Test set: Average loss: 0.0019, Accuracy: 8436/10000 (84.36%)

Epoch 70:
Loss=0.65577232837677 Batch_id=195 Accuracy=79.57: 100%|██████████| 196/196 [00:16<00:00, 11.84it/s]

Test set: Average loss: 0.0018, Accuracy: 8482/10000 (84.82%)

Epoch 71:
Loss=0.5376607775688171 Batch_id=195 Accuracy=80.07: 100%|██████████| 196/196 [00:16<00:00, 11.81it/s]

Test set: Average loss: 0.0018, Accuracy: 8520/10000 (85.20%)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**Validation Loss & Accuracy Graph**&#x1F537;

![image](https://user-images.githubusercontent.com/46663815/216041502-62769ac1-30cc-4665-adaf-7cd2f0a79580.png)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**_Sidenote - Depthwise Separable Convolution_**&#x1F537;

The main difference between 2D convolutions and Depthwise Convolution is that 2D convolutions are performed over all/multiple input channels, whereas in Depthwise convolution, each channel is kept separate.

Depthwise separable convolution does parameter reduction quite a lot. Let's see an example. At a given layer, let's suppose, the number of input channels are 128. Let's see the number of pamaters while we do a normal convolution with a kernel of 3 * 3 and we apply 256 such kernels. Then, for normal convolution case, the number of parameters would come as 3 * 3 * 128 * 256. In depthwise separable convolution, we apply number of convolutions equal to the number of channels. So, for depthwise convolution case, the number of parameters would come as (3 * 3 * 1 * 1 * 128 + 1 * 1 * 128 * 256). So, there would be parameter reduction factor as (3 * 3 * 128 * 256)/ (3 * 3 * 1 * 1 * 128 + 1 * 1 * 128 * 256) = 8.69.

![image](https://user-images.githubusercontent.com/46663815/216529563-49f11566-1a50-4d9a-8336-19b55bc8d824.png)

Even though we have 8 times reduction of parameters in case for depthwise separable convolutions, but accuracy drop than normal convolution would be few percentages only.Depthwise convolution is ideal for lightweight devices having less memory and compute, mobile phones could be an example. MobileNet version 1 and 2 are inspired by such technique.

But we have seen that in case where high accuracy is required, models with depthwise separable convolutions, tend to underfit a bit. And if the relative size of the model is very small, reducing parameters through depthwise separable convolution may lead to complete failure of the model.


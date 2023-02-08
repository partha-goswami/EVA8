&#x1F537;**Problem Definition**&#x1F537;

Task is to train resnet model for 20 epochs on CIFAR10 dataset and observe and plot loss curves, both for test and train datasets. Next task would be to identify 10 misclassified images. And we would have to show the GradCAM output on those 10 misclassified images.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


&#x1F537;**Repo**&#x1F537;

https://github.com/partha-goswami/pytorch-cifar

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**Model**&#x1F537;

https://github.com/partha-goswami/pytorch-cifar/blob/main/models/resnet.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**main.py**&#x1F537;

https://github.com/partha-goswami/pytorch-cifar/blob/main/main.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**utils.py**&#x1F537;

https://github.com/partha-goswami/pytorch-cifar/blob/main/utils.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**Training Log (epochs used: 20, best validation accuracy got: 89.15)**&#x1F537;

Epoch 1:
Loss=1.4874705076217651 Batch_id=195 Accuracy=39.03: 100%|██████████| 196/196 [01:24<00:00,  2.32it/s]

Test set: Average loss: 0.0073, Accuracy: 4161/10000 (41.61%)

Epoch 2:
Loss=1.065431833267212 Batch_id=195 Accuracy=54.61: 100%|██████████| 196/196 [01:16<00:00,  2.55it/s]

Test set: Average loss: 0.0055, Accuracy: 5451/10000 (54.51%)

Epoch 3:
Loss=0.9543948173522949 Batch_id=195 Accuracy=63.06: 100%|██████████| 196/196 [01:16<00:00,  2.55it/s]

Test set: Average loss: 0.0058, Accuracy: 5736/10000 (57.36%)

Epoch 4:
Loss=0.7508898377418518 Batch_id=195 Accuracy=69.85: 100%|██████████| 196/196 [01:16<00:00,  2.56it/s]

Test set: Average loss: 0.0045, Accuracy: 6432/10000 (64.32%)

Epoch 5:
Loss=0.5649515390396118 Batch_id=195 Accuracy=73.94: 100%|██████████| 196/196 [01:16<00:00,  2.55it/s]

Test set: Average loss: 0.0039, Accuracy: 6970/10000 (69.70%)

Epoch 6:
Loss=0.5525150299072266 Batch_id=195 Accuracy=76.89: 100%|██████████| 196/196 [01:17<00:00,  2.54it/s]

Test set: Average loss: 0.0036, Accuracy: 7103/10000 (71.03%)

Epoch 7:
Loss=0.6174148917198181 Batch_id=195 Accuracy=79.83: 100%|██████████| 196/196 [01:16<00:00,  2.55it/s]

Test set: Average loss: 0.0028, Accuracy: 7710/10000 (77.10%)

Epoch 8:
Loss=0.4799003005027771 Batch_id=195 Accuracy=82.18: 100%|██████████| 196/196 [01:16<00:00,  2.56it/s]

Test set: Average loss: 0.0025, Accuracy: 7916/10000 (79.16%)

Epoch 9:
Loss=0.6108114123344421 Batch_id=195 Accuracy=84.45: 100%|██████████| 196/196 [01:17<00:00,  2.54it/s]

Test set: Average loss: 0.0024, Accuracy: 8020/10000 (80.20%)

Epoch 10:
Loss=0.33017224073410034 Batch_id=195 Accuracy=86.22: 100%|██████████| 196/196 [01:16<00:00,  2.55it/s]

Test set: Average loss: 0.0025, Accuracy: 7983/10000 (79.83%)

Epoch 11:
Loss=0.48934707045555115 Batch_id=195 Accuracy=87.79: 100%|██████████| 196/196 [01:17<00:00,  2.54it/s]

Test set: Average loss: 0.0033, Accuracy: 7645/10000 (76.45%)

Epoch 12:
Loss=0.4562051296234131 Batch_id=195 Accuracy=89.48: 100%|██████████| 196/196 [01:17<00:00,  2.54it/s]

Test set: Average loss: 0.0020, Accuracy: 8471/10000 (84.71%)

Epoch 13:
Loss=0.5354408025741577 Batch_id=195 Accuracy=91.21: 100%|██████████| 196/196 [01:16<00:00,  2.56it/s]

Test set: Average loss: 0.0024, Accuracy: 8322/10000 (83.22%)

Epoch 14:
Loss=0.17979788780212402 Batch_id=195 Accuracy=92.84: 100%|██████████| 196/196 [01:16<00:00,  2.55it/s]

Test set: Average loss: 0.0019, Accuracy: 8593/10000 (85.93%)

Epoch 15:
Loss=0.15545208752155304 Batch_id=195 Accuracy=94.32: 100%|██████████| 196/196 [01:16<00:00,  2.55it/s]

Test set: Average loss: 0.0023, Accuracy: 8472/10000 (84.72%)

Epoch 16:
Loss=0.07985072582960129 Batch_id=195 Accuracy=95.60: 100%|██████████| 196/196 [01:16<00:00,  2.55it/s]

Test set: Average loss: 0.0017, Accuracy: 8805/10000 (88.05%)

Epoch 17:
Loss=0.03720656782388687 Batch_id=195 Accuracy=96.67: 100%|██████████| 196/196 [01:16<00:00,  2.56it/s]

Test set: Average loss: 0.0017, Accuracy: 8837/10000 (88.37%)

Epoch 18:
Loss=0.018214857205748558 Batch_id=195 Accuracy=97.63: 100%|██████████| 196/196 [01:16<00:00,  2.55it/s]

Test set: Average loss: 0.0017, Accuracy: 8894/10000 (88.94%)

Epoch 19:
Loss=0.02978559397161007 Batch_id=195 Accuracy=98.08: 100%|██████████| 196/196 [01:16<00:00,  2.55it/s]

Test set: Average loss: 0.0016, Accuracy: 8915/10000 (89.15%)

Epoch 20:
Loss=0.040735505521297455 Batch_id=195 Accuracy=98.24: 100%|██████████| 196/196 [01:16<00:00,  2.56it/s]

Test set: Average loss: 0.0017, Accuracy: 8904/10000 (89.04%)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

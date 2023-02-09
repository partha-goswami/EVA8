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

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;## Training Log (model: resnet18, epochs used: 20, best validation accuracy got: 92.0)&#x1F537;

Epoch 1:
Loss=1.1870989799499512 Batch_id=195 Accuracy=42.74: 100%|██████████| 196/196 [00:51<00:00,  3.80it/s]

Test set: Average loss: 0.0053, Accuracy: 5220/10000 (52.20%)

Epoch 2:
Loss=1.1822130680084229 Batch_id=195 Accuracy=57.46: 100%|██████████| 196/196 [00:43<00:00,  4.51it/s]

Test set: Average loss: 0.0050, Accuracy: 5929/10000 (59.29%)

Epoch 3:
Loss=0.9037604331970215 Batch_id=195 Accuracy=66.15: 100%|██████████| 196/196 [00:43<00:00,  4.49it/s]

Test set: Average loss: 0.0044, Accuracy: 6355/10000 (63.55%)

Epoch 4:
Loss=1.023028016090393 Batch_id=195 Accuracy=71.46: 100%|██████████| 196/196 [00:43<00:00,  4.50it/s]

Test set: Average loss: 0.0044, Accuracy: 6486/10000 (64.86%)

Epoch 5:
Loss=0.7488135695457458 Batch_id=195 Accuracy=75.29: 100%|██████████| 196/196 [00:43<00:00,  4.50it/s]

Test set: Average loss: 0.0028, Accuracy: 7599/10000 (75.99%)

Epoch 6:
Loss=0.6292551159858704 Batch_id=195 Accuracy=78.11: 100%|██████████| 196/196 [00:43<00:00,  4.47it/s]

Test set: Average loss: 0.0026, Accuracy: 7821/10000 (78.21%)

Epoch 7:
Loss=0.7022161483764648 Batch_id=195 Accuracy=80.53: 100%|██████████| 196/196 [00:43<00:00,  4.49it/s]

Test set: Average loss: 0.0022, Accuracy: 8102/10000 (81.02%)

Epoch 8:
Loss=0.4395454525947571 Batch_id=195 Accuracy=82.60: 100%|██████████| 196/196 [00:43<00:00,  4.49it/s]

Test set: Average loss: 0.0026, Accuracy: 7836/10000 (78.36%)

Epoch 9:
Loss=0.4539540708065033 Batch_id=195 Accuracy=84.42: 100%|██████████| 196/196 [00:43<00:00,  4.49it/s]

Test set: Average loss: 0.0023, Accuracy: 7997/10000 (79.97%)

Epoch 10:
Loss=0.34765860438346863 Batch_id=195 Accuracy=85.73: 100%|██████████| 196/196 [00:43<00:00,  4.49it/s]

Test set: Average loss: 0.0021, Accuracy: 8308/10000 (83.08%)

Epoch 11:
Loss=0.3739401698112488 Batch_id=195 Accuracy=87.41: 100%|██████████| 196/196 [00:43<00:00,  4.47it/s]

Test set: Average loss: 0.0021, Accuracy: 8337/10000 (83.37%)

Epoch 12:
Loss=0.4225074350833893 Batch_id=195 Accuracy=88.32: 100%|██████████| 196/196 [00:43<00:00,  4.49it/s]

Test set: Average loss: 0.0015, Accuracy: 8753/10000 (87.53%)

Epoch 13:
Loss=0.3384374678134918 Batch_id=195 Accuracy=89.81: 100%|██████████| 196/196 [00:43<00:00,  4.48it/s]

Test set: Average loss: 0.0015, Accuracy: 8825/10000 (88.25%)

Epoch 14:
Loss=0.1865644007921219 Batch_id=195 Accuracy=91.35: 100%|██████████| 196/196 [00:43<00:00,  4.50it/s]

Test set: Average loss: 0.0016, Accuracy: 8703/10000 (87.03%)

Epoch 15:
Loss=0.455554336309433 Batch_id=195 Accuracy=92.53: 100%|██████████| 196/196 [00:43<00:00,  4.49it/s]

Test set: Average loss: 0.0015, Accuracy: 8844/10000 (88.44%)

Epoch 16:
Loss=0.17733941972255707 Batch_id=195 Accuracy=93.85: 100%|██████████| 196/196 [00:43<00:00,  4.47it/s]

Test set: Average loss: 0.0014, Accuracy: 8994/10000 (89.94%)

Epoch 17:
Loss=0.1845535784959793 Batch_id=195 Accuracy=94.70: 100%|██████████| 196/196 [00:43<00:00,  4.48it/s]

Test set: Average loss: 0.0012, Accuracy: 9092/10000 (90.92%)

Epoch 18:
Loss=0.18869087100028992 Batch_id=195 Accuracy=95.89: 100%|██████████| 196/196 [00:43<00:00,  4.50it/s]

Test set: Average loss: 0.0011, Accuracy: 9157/10000 (91.57%)

Epoch 19:
Loss=0.13974884152412415 Batch_id=195 Accuracy=96.46: 100%|██████████| 196/196 [00:43<00:00,  4.48it/s]

Test set: Average loss: 0.0011, Accuracy: 9200/10000 (92.00%)

Epoch 20:
Loss=0.05817868560552597 Batch_id=195 Accuracy=96.71: 100%|██████████| 196/196 [00:43<00:00,  4.48it/s]

Test set: Average loss: 0.0011, Accuracy: 9198/10000 (91.98%)


----------------------------------------------------------------------------------------------------------------------------------------------------------------------

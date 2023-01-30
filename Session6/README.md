&#x1F537;**Problem Definition**


While using Depthwise separable convolution and dilated convolution blocks, we would have a design a neural network that achieves at least 85 % accuracy on CIFAR10 dataset. Total number of model parameters should be lesser than 200k, moreover we should be using GAP as one of the layers. The receptive field should be above 44. We would have to use a code that has the following basic blocks as convolution block1, convolution block2, convolution block3, convolution block 4, and output.

&#x1F537;**Model**

https://github.com/partha-goswami/EVA8/blob/main/Session6/Model/model.py

&#x1F537;**Model Parameters and Final RF**

**Model Parameters**: 97284

**Final Receptive Fiield at the last layer**: 109

_Input, output and RF calculations are done at each step and is detailed at model.py file above. I have used both dilution and depthwise separable convolution._

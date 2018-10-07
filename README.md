# mnist_classify-handwriten-NN-network

This project is refered from the [cs231n-assignment](https://github.com/czy97/cs231n) code I finished in September 2018.

## Require：
- python 3.6  
- numpy  
- download mnist data in the codes/data/

## Function:
- Define arbitrary hiddens layers as you want  
- Choose activation function between sigmoid and relu  
- BatchNorm  
- Dropout  
- Model storing(FullyConnectedNet.storeModel())  
- Model loading(FullyConnectedNet.loadModel())  
- Choose different update rules among sgd/sgd_momentum/rmsprop/adam  
- Seperated model definition module(codes.classifiers.fc_net.FullyConnectedNet) and updating module(codes.solver.Solver)

## Demo
mnist_classification.ipynb

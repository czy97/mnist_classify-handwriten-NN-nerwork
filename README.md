# mnist_classify-handwriten-NN-network

This project is refered from the [cs231n-assignment](https://github.com/czy97/cs231n) code I finished in August 2018.

## Require：
- python 3.6  
- numpy  
- download mnist data and store in the codes/data/

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

## Postscript
The bestParams.pkl in the params directory I didn't upload here.  
If you need email me to chenzhengyang117@gmail.com

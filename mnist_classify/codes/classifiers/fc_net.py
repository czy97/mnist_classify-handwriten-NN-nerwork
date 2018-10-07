import numpy as np

from codes.layers import *
from codes.layer_utils import *

class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None,activationFunc = 'relu'):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    if(activationFunc == 'relu'):
      self.activation_forward = relu_forward
      self.activation_backward = relu_backward
    if(activationFunc == 'sigmoid'):
      self.activation_forward = sigmoid_forward
      self.activation_backward = sigmoid_backward

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    self.batchNormPara= {}
    hidden_dims.insert(0,input_dim)
    hidden_dims.append(num_classes)
    
    for i in range(self.num_layers):
      W_name = 'W' + str(i+1)
      b_name = 'b' + str(i+1)
      self.params[W_name] = weight_scale * np.random.randn(hidden_dims[i], hidden_dims[i+1])
      self.params[b_name] = np.zeros(hidden_dims[i+1])
    if self.use_batchnorm:
      for i in range(self.num_layers - 1):
        gamma_name = 'gamma' + str(i+1)
        beta_name = 'beta' + str(i+1)
        self.params[gamma_name] = np.ones(hidden_dims[i+1])
        self.params[beta_name] = np.zeros(hidden_dims[i+1])
    #print self.params.keys()
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    output = None
    out_activation = None
    out_batchnorm = None
    out_dropout = None
    cache = {}
    cache_activation = {}
    cache_batchnorm = {}
    cache_dropout = {}
    
    


    out_unit= X
    for i in range(self.num_layers - 1):
      W_name = 'W' + str(i+1)
      b_name = 'b' + str(i+1)
      

      output_a,cache[i+1] = affine_forward(out_unit, self.params[W_name], self.params[b_name])
      tmpOutput = output_a
      if self.use_batchnorm:
        gamma_name = 'gamma' + str(i+1)
        beta_name = 'beta' + str(i+1)
   
        out_batchnorm, cache_batchnorm[i+1] = batchnorm_forward(output_a, self.params[gamma_name], self.params[beta_name], self.bn_params[i])
        tmpOutput = out_batchnorm
      out_activation,cache_activation[i+1] = self.activation_forward(tmpOutput)
      out_unit = out_activation
      if self.use_dropout:
        out_unit, cache_dropout[i+1] = dropout_forward(out_unit, self.dropout_param)
      
      
    W_name = 'W' + str(self.num_layers)
    b_name = 'b' + str(self.num_layers)
    scores,cache[self.num_layers] = affine_forward(out_unit, self.params[W_name], self.params[b_name])

    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    N = X.shape[0]
    expTmp = np.exp(scores)
    possTmp = expTmp/(np.sum(expTmp,1).reshape(N,1))
    regularLoss = sum(sum(self.params['W1']**2)) + sum(sum(self.params['W2']**2))
    labelArray = np.zeros_like(possTmp)#label where the possibity should be zero
    tmpList = np.array(range(N))
    labelArray[tmpList,y] = 1
    loss = sum(sum(-np.log(possTmp) * labelArray))/N + self.reg * regularLoss *0.5

    dout = possTmp.copy()
    dout -= labelArray
    
    W_name = 'W' + str(self.num_layers)
    b_name = 'b' + str(self.num_layers)
 
    dunit,grads[W_name], grads[b_name] = affine_backward(dout, cache[self.num_layers])
    grads[W_name] = grads[W_name]/N + self.reg * self.params[W_name]
    grads[b_name] = grads[b_name]/N

    for i in range(self.num_layers-1,0,-1):
      W_name = 'W' + str(i)
      b_name = 'b' + str(i)
      if self.use_dropout:
        dunit = dropout_backward(dunit, cache_dropout[i])
      dout_tmp = self.activation_backward(dunit, cache_activation[i])
      if self.use_batchnorm:
        gamma_name = 'gamma' + str(i)
        beta_name = 'beta' + str(i)
        dout_tmp, dgamma, dbeta = batchnorm_backward(dout_tmp, cache_batchnorm[i])
        grads[gamma_name] = dgamma.copy() 
        grads[beta_name] = dbeta.copy()
      dunit, grads[W_name], grads[b_name] = affine_backward(dout_tmp, cache[i])
      grads[W_name] = grads[W_name]/N + self.reg * self.params[W_name]
      grads[b_name] = grads[b_name]/N


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
  def storeModel(self,filename = 'bestParams'):
    prefix = 'params/'
    filepath = prefix + filename + '.pkl'

    import pickle
    params = {}
    params['weight_params'] = self.params
    params['bn_params'] = self.bn_params
    with open(filepath, 'wb') as f:
      pickle.dump(params, f)
  def loadModel(self,filename = 'bestParams'):
    prefix = 'params/'
    filepath = prefix + filename + '.pkl'

    import pickle
    with open(filepath, 'rb') as f:
      params = pickle.load(f)
    self.params = params['weight_params']
    self.bn_params = params['bn_params']
  def predict(self,X):
    return np.argmax(self.loss(X), axis=1)
  def getAcc(self,X,y):
    y_pred = self.predict(X)
    acc = (y_pred == y).mean()
    return acc

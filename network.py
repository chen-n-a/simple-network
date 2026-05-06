from fileFunctions import *
import numpy as np
from matlabFunctions import XTrain, yTrain, XTest, yTest, label
import time

L = 6 #layer count, not including the input layer
n = [784, 3136, 3136, 2352, 1568, 248, 62] #decided on these number of neuros because they add up to about 10000

def networkShape():
  '''Returns the shape of the network as a tuple (L, n) where L is the number of layers and n is a list of the number of neurons in each layer.
  
  Returns
  -------
  L : int
      The number of layers in the network, not including the input layer.
  n : list
      A list of the number of neurons in each layer, including the input layer.
  '''
      
  global L, n
  return L, n 

def initiliseNetworkValues(n : list):
  '''Initializes the weights and biases of the network using He initialization for the weights and zeros for the biases. The initialized values are saved to files for later use.
  
  Parameters
  ----------
  n : list
      A list of the number of neurons in each layer, including the input layer.
  '''
  
  global weights, biases
  
  #clears files before using them jussst incase, thought saving intial values separately from the training values would be a good idea to prevent any accidental overwriting of the initial values, and to make it easier to reset the network to its initial state if needed
  clearFile('initialWeights')
  clearFile('initialBiases')
  clearFile('metadata')
  
  weights, biases, metadata = {}, {}, {} #storing as dictionaries because hdf5 files
  for i in range(1,7):
    weights[f'W{i}'] = np.random.randn(n[i], n[i-1]).astype(np.float32) * np.sqrt(2.0 / n[i-1])
    biases[f'b{i}'] = np.zeros((n[i], 1), dtype=np.float32)
    #Using He initialisation to prevent dissapearing gradients/excessively large ones later

  metadata = {'t': np.array([0], dtype=np.int32),
              'epochs': np.array([0], dtype=np.int32)
  } # initializing the timestep for adam to 0, and saving it in a dictionary to be stored in a file and updated across batches and epochs
  '''
  for value in weights:
      print(f"Weights for layer {value} shape:", weights[f'{value}'].shape) 
  
  for value in biases:
      print(f"bias for layer {value} shape:", biases[f'{value}'].shape)
  # using these to check for size
  '''
  saveToFile(weights, 'initialWeights')
  saveToFile(biases, 'initialBiases')
  saveToFile(metadata, 'metadata')

def initAdam(weights, biases):
  '''Initializes the Adam optimizer's moment estimates (mW, mB, vW, vB) to zeros and saves them to files for later use.
  
  Parameters
  ----------
  weights : dict
      A dictionary containing the weights of the network.
  biases : dict
      A dictionary containing the biases of the network.
  '''
  
  global mW, mB, vW, vB, t
  
  #same as initialiseNetworkValues()
  clearFile('adamMW')
  clearFile('adamMB')
  clearFile('adamVW')
  clearFile('adamVB')
  clearFile('adamt')
  
  mW, mB, vW, vB = {}, {}, {}, {} #hdf5 strikes again
  for i in range(1, 7): #learnt this as a good way to not type something out 6 times
    vW[f'W{i}'] = np.zeros_like(weights[f'W{i}'])
    vB[f'b{i}'] = np.zeros_like(biases[f'b{i}'])
    mW[f'W{i}'] = np.zeros_like(weights[f'W{i}'])
    mB[f'b{i}'] = np.zeros_like(biases[f'b{i}'])
    
  saveToFile(mW, 'adamMW')
  saveToFile(mB, 'adamMB')
  saveToFile(vW, 'adamVW')
  saveToFile(vB, 'adamVB')
  
def prepare_data(XTrain, yTrain, batchSize) -> tuple[float, float, int, int]:
  '''Prepares the training data by normalizing the pixel values, converting the labels to one-hot encoding, and calculating the number of batches based on the batch size.
  
  Parameters
  ----------
  XTrain : numpy.ndarray
      The training images.
  yTrain : numpy.ndarray
      The training labels.
  batchSize : int
      The size of each batch.
  
  Returns
  -------
  A0 : numpy.ndarray
      The normalized input data.
  Y : numpy.ndarray
      The one-hot encoded labels.
  m : int
      The number of training samples.
  numBatches : int
      The number of batches.
  '''
  
  classNumber = 62 #number of classes in dataset
  X = XTrain.T.astype(np.float32) / 255.0 #normalizing the pixel values to be between 0 and 1, and transposing the data so that each column is a sample and each row is the same pixel in the photo, this makes it easier to perform matrix operations during feedforward and backpropagation
  y = yTrain.flatten().astype(int) #flattening the labels to be a 1D array and converting to integers for indexing, this makes it easier to convert the labels to one-hot encoding and to calculate the accuracy later on
  m = y.shape[0] #number of training samples, this is used to calculate the cost and to determine the number of batches based on the batch size, and to ensure the data has been prepared correctly by checking it against the original number of samples in the dataset
  Y = np.zeros((classNumber, m), dtype=np.float32)   #initializing the one-hot encoded labels as a matrix of zeros with dimensions (number of classes, number of samples), this makes it easier to perform matrix operations during feedforward and backpropagation, and to calculate the cost and accuracy later on
  A0 = X #setting A0 to be the input data, this is used as the input to the first layer of the network during feedforward
  Y[y, np.arange(m)] = 1.0 #converting the labels to one-hot encoding by setting the appropriate element in each column to 1 based on the label value, this makes it easier to calculate the cost and accuracy later on by allowing us to compare the predicted probabilities with the true labels in a matrix form
  numBatches = int(np.ceil(m / batchSize))
  #expecting (784, 697932) (62, 697932) a small number(thousands) 0 to 61   checks to make sure the data has been prepared and formated properly (row to collum)
  print(f"Images Shape: {X.shape} (Pixels, Samples)")
  print(f"Labels Shape: {Y.shape} (Classes, Samples)")
  print(f"Total Batches: {numBatches}")
  print(f"Label range: {np.min(y)} to {np.max(y)}")
  return A0, Y, m, numBatches

def sigmoid(z : float) -> float:
  '''Calculates the sigmoid activation function for a given input z.
  
  Parameters
  ----------
  z : float
      The input value for which to calculate the sigmoid activation.
  
  Returns
  -------
  float
      The output of the sigmoid activation function.
  '''
  return 1 / (1 + np.exp(-1 * z))

def silu(z : float) -> tuple[float, float]:
  '''Calculates the SiLU (Sigmoid Linear Unit) activation function for a given input z, as well as the sigmoid value for use in backpropagation. 
  This means the sigmoid value is only calculated once
  
  Parameters
  ----------
  z : float
      The input value for which to calculate the SiLU activation.
      
  Returns
  -------
  A : float
      The output of the SiLU activation function.
  s : float
      The sigmoid value for the given input
  '''
  s = sigmoid(z)
  A = z * s
  return A, s

def siluDeriv(z : float, s : float) -> float:
  '''Calculates the derivative of the SiLU activation function for a given input z and sigmoid value s, which is used in backpropagation to calculate the gradients for the weights and biases.
  
  Parameters
  ----------
  z : float
      The input value for which to calculate the derivative.
  s : float
      The sigmoid value for the given input.
  
  Returns
  -------
  float
      The derivative of the SiLU activation function.
  '''
  
  return (z * s) + (s * (1 - (z * s)))

def softmax(z : float) -> float:
  '''Calculates the softmax activation function for a given input z, which is used in the output layer of the network to convert the raw output values into probabilities that sum to 1.
  Softmax is used in the output layer because it is a multi-class classification problem, and softmax allows us to interpret the output as probabilities for each class.
  Softmax works by exponentiating the input values to ensure they are positive, and then normalizing them by dividing by the sum of the exponentiated values to ensure they sum to 1 a bit like partial pressures
  
  Parameters
  ----------
  z : float
      The input value for which to calculate the softmax activation.
  
  Returns
  -------
  float
      The output of the softmax activation function.
  '''
  
  zMax = np.max(z, axis=0, keepdims=True) 
  zShift = np.exp(z - zMax) 
  return zShift / np.sum(zShift, axis=0, keepdims=True)

def cost(yHat : float, Y : float) -> float:
  '''Calculates the cross-entropy cost function for the predicted probabilities yHat and the true labels Y, which is used to evaluate the performance of the network and to guide the optimization process during training.
  The cost function is calculated as the negative average of the element-wise product of the true labels and the logarithm of the predicted probabilities, which penalizes incorrect predictions more heavily and encourages the network to output probabilities that are close to the true labels.
  
  Parameters
  ----------
  yHat : float
      The predicted probabilities.
  Y : float
      The true labels.
  
  Returns
  -------
  cost : float
      The output of the cross-entropy cost function.
  '''
  
  m = Y.shape[1]
  cost = - (1/m) * np.sum(Y * np.log(yHat + 1e-12)) #negative average is used so cost is being minimised
  return cost

def feedForword(A0 : float, weights : dict, biases : dict) -> dict:
  '''Performs the feedforward pass through the network, calculating the activations for each layer based on the input data A0, the weights, and the biases. The activations are calculated using matrix multiplication followed by the application of the activation functions (SiLU for hidden layers and softmax for the output layer). The intermediate values (Z and s) are stored in a cache for use in backpropagation.
  
  Parameters
  ----------
  A0 : float
      The input data for the first layer.
  weights : dict
      A dictionary containing the weights for each layer.
  biases : dict
      A dictionary containing the biases for each layer.
  
  Returns
  -------
  cache : dict
      A dictionary containing the cached values for backpropagation.
  '''
  Z1 = weights['W1'] @ A0 + biases['b1']
  A1, s1 = silu(Z1)

  Z2 = weights['W2'] @ A1 + biases['b2']
  A2, s2 = silu(Z2)

  Z3 = weights['W3'] @ A2 + biases['b3']
  A3, s3 = silu(Z3)
  
  Z4 = weights['W4'] @ A3 + biases['b4']
  A4, s4 = silu(Z4)
  
  Z5 = weights['W5'] @ A4 + biases['b5']
  A5, s5 = silu(Z5)
  
  Z6 = weights['W6'] @ A5 + biases['b6']
  A6 = softmax(Z6)

  cache = {
    'Z1': Z1, 's1': s1, 'A1': A1,
    'Z2': Z2, 's2': s2, 'A2': A2,
    'Z3': Z3, 's3': s3, 'A3': A3,
    'Z4': Z4, 's4': s4, 'A4': A4,
    'Z5': Z5, 's5': s5, 'A5': A5,
    'Z6': Z6, 'A6': A6,
  }

  return cache

def backprop(weights,cache, X, Y) -> tuple[dict, dict]:
  '''Performs the backpropagation pass through the network, calculating the gradients for the weights and biases based on the cached values from the feedforward pass, the input data X, and the true labels Y. The gradients are calculated using chain rule, starting from the output layer and propagating backwards through the hidden layers. The gradients are returned as dictionaries for use in the optimization step.
  
  Parameters
  ----------
  weights : dict
      A dictionary containing the weights for each layer.
  cache : dict
      A dictionary containing the cached values from the feedforward pass.
  X : numpy.ndarray
      The input data.
  Y : numpy.ndarray
      The true labels.
  
  Returns
  -------
  weightsGradients : dict
      A dictionary containing the gradients for the weights.
  biasGradients : dict
      A dictionary containing the gradients for the biases.
  '''
  
  #bassicaly the opposite of feedforward, if silu then silu derive etc
  m = X.shape[1] #number of samples in the batch, used to calculate the average gradients for the weights and biases, which helps to stabilize the training process and prevent excessively large updates to the weights and biases
  dZ6 = cache['A6'] - Y
  dW6 = (1/m) * (dZ6 @ cache['A5'].T)
  db6 = (1/m) * np.sum(dZ6, axis=1, keepdims=True)

  dA5 = weights['W6'].T @ dZ6 #calculating activation gradients, this starts the whole process
  dz5 = dA5 * siluDeriv(cache['Z5'], cache['s5']) #calculating Z gradients (the input to the activation function) 
  dW5 = (1/m) * (dz5 @ cache['A4'].T) #calculating weight gradients by multiplying the Z gradients with the activations from the previous layer, and averaging over the batch
  db5 = (1/m) * np.sum(dz5, axis=1, keepdims=True) #calculating bias gradients
  
  dA4 = weights['W5'].T @ dz5
  dz4 = dA4 * siluDeriv(cache['Z4'], cache['s4'])
  dW4 = (1/m) * (dz4 @ cache['A3'].T)
  db4 = (1/m) * np.sum(dz4, axis=1, keepdims=True)
  
  dA3 = weights['W4'].T @ dz4
  dz3 = dA3 * siluDeriv(cache['Z3'], cache['s3'])
  dW3 = (1/m) * (dz3 @ cache['A2'].T)
  db3 = (1/m) * np.sum(dz3, axis=1, keepdims=True)
  
  dA2 = weights['W3'].T @ dz3
  dz2 = dA2 * siluDeriv(cache['Z2'], cache['s2'])
  dW2 = (1/m) * (dz2 @ cache['A1'].T)
  db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
  
  dA1 = weights['W2'].T @ dz2
  dz1 = dA1 * siluDeriv(cache['Z1'], cache['s1'])
  dW1 = (1/m) * (dz1 @ X.T)
  db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

  weightsGradients = {
    'dW1': dW1, 'dW2': dW2, 'dW3': dW3, 'dW4': dW4, 'dW5': dW5, 'dW6': dW6
  }
  biasGradients = {
    'db1': db1, 'db2': db2, 'db3': db3, 'db4': db4, 'db5': db5, 'db6': db6
  }
  
  return weightsGradients, biasGradients

def adam(weights : dict, biases : dict, weightsGradients : dict, biasGradients : dict, mW : dict, mB : dict, vW : dict, vB : dict, learningRate : float, batch : int) -> tuple[dict, dict, dict, dict, dict, dict]:
  '''Performs the Adam optimization algorithm to update the weights and biases of the network based on the calculated gradients, the moment estimates (mW, mB, vW, vB), the learning rate, and the current batch number. The moment estimates are updated using exponential moving averages of the gradients and their squares, and bias-corrected estimates are used to calculate the parameter updates. The updated weights, biases, and moment estimates are returned for use in the next iteration of training.
  beta1 and 2 are decay rates for moment estimates and epsilon is to prevent div by zero, all are common hyperparameters for adam
  
  Parameters
  ----------
  weights : dict
      A dictionary containing the weights for each layer.
  biases : dict
      A dictionary containing the biases for each layer.
  weightsGradients : dict
      A dictionary containing the gradients for the weights.
  biasGradients : dict
      A dictionary containing the gradients for the biases.
  mW : dict
      A dictionary containing the first moment estimates for the weights.
  mB : dict
      A dictionary containing the first moment estimates for the biases.
  vW : dict
      A dictionary containing the second moment estimates for the weights.
  vB : dict
      A dictionary containing the second moment estimates for the biases.
  learningRate : float
      The learning rate for the Adam optimization algorithm.
  batch : int
      The current batch number.

  Returns
  -------
  weights : dict
      The updated weights.
  biases : dict
      The updated biases.
  mW : dict
      The updated first moment estimates for the weights.
  mB : dict
      The updated first moment estimates for the biases.
  vW : dict
      The updated second moment estimates for the weights.
  vB : dict
      The updated second moment estimates for the biases.
  '''
  
  beta1 = 0.9
  beta2 = 0.999
  epsilon = 1e-8
  
  for i in range(1, 7):
    #first moment estimates are a moving average of the gradients, second moment estimates are a moving average of the squared gradients
    mW[f'W{i}'] = beta1 * mW[f'W{i}'] + (1 - beta1) * weightsGradients[f'dW{i}'] 
    mB[f'b{i}'] = beta1 * mB[f'b{i}'] + (1 - beta1) * biasGradients[f'db{i}']
    
    vW[f'W{i}'] = beta2 * vW[f'W{i}'] + (1 - beta2) * (weightsGradients[f'dW{i}'] ** 2)
    vB[f'b{i}'] = beta2 * vB[f'b{i}'] + (1 - beta2) * (biasGradients[f'db{i}'] ** 2)
    
    #bigger the batch the smaller the updates, more trust in the data hopefully less noise
    mWCorrected = mW[f'W{i}'] / (1 - beta1 ** batch)
    mBCorrected = mB[f'b{i}'] / (1 - beta1 ** batch)
    
    vWCorrected = vW[f'W{i}'] / (1 - beta2 ** batch)
    vBCorrected = vB[f'b{i}'] / (1 - beta2 ** batch)
    
    #adam update rule, dividing the learning rate by the square root of the second moment estimate (plus epsilon), better than stochastic gradient descent
    weights[f'W{i}'] -= learningRate * mWCorrected / (np.sqrt(vWCorrected) + epsilon) 
    biases[f'b{i}'] -= learningRate * mBCorrected / (np.sqrt(vBCorrected) + epsilon)
    
  return weights, biases, mW, mB, vW, vB

def accuracy(yHat : float, Y : float) -> float:
  '''Calculates the accuracy of the predictions by comparing the predicted probabilities yHat with the true labels Y.
  
  Parameters
  ----------
  yHat : float
      The predicted probabilities.
  Y : float
      The true labels.

  Returns
  -------
  float
      The accuracy of the predictions.
  '''
  
  predictions = np.argmax(yHat, axis=0)
  labels = np.argmax(Y, axis=0)
  return np.mean(predictions == labels)*100

def testModel(XTest : np.ndarray, yTest : np.ndarray, weights : dict, biases : dict, batchSize : int):
  '''Tests the trained model on the test set by performing a feedforward pass and calculating the accuracy of the predictions.
  
  Parameters
  ----------
  XTest : np.ndarray
      The test data.
  yTest : np.ndarray
      The true labels for the test data.
  weights : dict
      A dictionary containing the weights for each layer.
  biases : dict
      A dictionary containing the biases for each layer.
  batchSize : int
      The size of each batch.
  '''
  
  tA0, tY, tm, tnumBatches = prepare_data(XTest, yTest, batchSize)
  totalCorrect = 0

  for i in range(0, tm, batchSize):
    #no need to backprop, much faster process, and larger batchsize as less maths per image is required
    tXbatch = tA0[:, i:i+batchSize]
    tYbatch = tY[:, i:i+batchSize]
    tcache = feedForword(tXbatch, weights, biases)
    tYhat = tcache['A6']
    predictions = np.argmax(tYhat, axis=0)
    labels = np.argmax(tYbatch, axis=0)
    totalCorrect += np.sum(predictions == labels)
  testAccuracy = (totalCorrect / tm) * 100
  print(f'Test Accuracy: {testAccuracy:.2f}%')
    
def train(learningRate : float, epochs : int, X : np.ndarray, Y : np.ndarray, batchSize : int):
  '''Trains the neural network using the Adam optimization algorithm, performing feedforward and backpropagation for each batch of training data, and updating the weights and biases accordingly. The training process is monitored by printing the cost and accuracy for each batch, as well as the average cost for each epoch. The model is also tested on the test set after each epoch to monitor generalization performance and check for overfitting.
  
  Parameters
  ----------
  learningRate : float
      The learning rate for the Adam optimizer.
  epochs : int
      The number of epochs to train the model.
  X : np.ndarray
      The training data.
  Y : np.ndarray
      The true labels for the training data.
  batchSize : int
      The size of each batch.
  '''
  
  global weights, biases
  epochStartTime = time.time()

  mW = loadData('adamMW')
  mB = loadData('adamMB')
  vW = loadData('adamVW')
  vB = loadData('adamVB')
  metadata = loadData('metadata')
  t = metadata['t'][0]
  
  if int(metadata['epochs'][0]) == 0: #gotta load this to start training
    print('Initialising network values...')
    weights = loadData('initialWeights')
    biases = loadData('initialBiases')  
  else: #gotta try load this so the network isnt diagnosed with amnesia
    weights = loadData('networkweights')
    biases = loadData('networkbiases')  
  
  print("Network files loaded successfully.")
  
  for epoch in range(int(metadata['epochs'][0]), epochs): #this helps the program remember what epoch was last saved
    startTime = time.time()
    epochCost = 0
    metadata = loadData('metadata') # loading the metadata at the start of each epoch to ensure the timestep and epoch number are updated correctly across batches and epochs
    t = int(metadata['t'][0]) # loading the timestep for adam from the file, to ensure it is updated correctly across batches and epochs
    permutation = np.random.permutation(X.shape[1])
    xShuffled = X[:, permutation]
    yShuffled = Y[:, permutation]
    
    for i in range(0, X.shape[1], batchSize):
      t += 1 # incrementing the timestep for adam at the start of each batch, to ensure it is updated correctly across batches and epochs
      xBatch = xShuffled[:, i:i+batchSize]
      yBatch = yShuffled[:, i:i+batchSize]
      
      #follows the pattern of feedforward, find cost, backprop, adam update and repeat
      cache = feedForword(xBatch, weights, biases)
      yHat = cache['A6']
      
      neworkCost = cost(yHat, yBatch)
      epochCost += neworkCost
      
      weightsGradients, biasGradients = backprop(weights, cache, xBatch, yBatch)
      
      weights, biases, mW, mB, vW, vB = adam(weights, biases, weightsGradients, biasGradients, mW, mB, vW, vB, learningRate, t)
            
      batchAccuracy = accuracy(yHat, yBatch)
      
      if i % (batchSize * 200) == 0: #keep an eye on the program 'bad program the cost is going up'
        print(f'Epoch {epoch + 1}, Batch {i // batchSize + 1}, Cost: {neworkCost:.4f}, Accuracy: {batchAccuracy:.4f}') 
        
    metadata['t'][0] = t #very important to save t for epoch so adam works, unsaved t means adam never adjusts the learning rate
    metadata['epochs'][0] = epoch + 1 
    
    #important to clear the current weights and biases first unless you want to lobotomise the network
    clearFile('networkweights')
    clearFile('networkbiases')
    saveToFile(weights, 'networkweights')
    saveToFile(biases, 'networkbiases')
    saveToFile(mW, 'adamMW') #also very important to save these unless you want adam to forget everything
    saveToFile(mB, 'adamMB')
    saveToFile(vW, 'adamVW')
    saveToFile(vB, 'adamVB')
    saveToFile(metadata, 'metadata') #this also helps adam not have amnesia
    duration = time.time() - startTime #nice to track time
    avgCost = epochCost / (X.shape[1] / batchSize)
    print(f'Epoch {epoch + 1} completed in {duration:.2f} seconds. Average Cost: {avgCost:.4f}.')
    testModel(XTest, yTest, weights, biases, batchSize*4) # testing the model on the test set after each epoch to monitor generalisation performance and check for overfitting
  
  totalDuration = time.time() - epochStartTime
  print(f'Training completed in {totalDuration:.2f} seconds.')
  print(f'Total Epochs: {epochs}, Learning Rate: {learningRate}')

def predict(X : np.ndarray) -> np.ndarray :
  '''Predicts the class labels for the given input data X using the trained weights and biases of the network. The function performs a feedforward pass through the network and returns the predicted class labels based on the output probabilities.
  
  Parameters
  ----------
  X : np.ndarray
      The input data for which to make prediction.

  Returns
  -------
  labeled : str
      The predicted class labels for the input data.
  '''
  
  weights = loadData('networkweights')
  biases = loadData('networkbiases') 
  A0, y, m, numBatches = prepare_data(X, np.array([1]), 1) #preparing the data in the same way as the training and test data, but with a batch size of 1 since we are only predicting for one sample, this ensures the input data is normalized and formatted correctly for the feedforward pass
  cache = feedForword(A0, weights, biases)
  yHat = cache['A6']
  prediction = np.argmax(yHat, axis=0)
  print(prediction)
  labeled = label[int(prediction[0])]
  return labeled
  
def initialiseNetwork(n : list):
  '''Initialises the network.
  
  Parameters
  ----------
  n : list
      A list containing the number of neurons in each layer.
  '''
  
  initiliseNetworkValues(n)
  initAdam(weights, biases)

def main(batchSize : int, epochs : int):
  '''handy way to run the whole training process'''
  global A0, Y, m, numBatches
  A0, Y, m, numBatches = prepare_data(XTrain, yTrain, batchSize)
  train(learningRate=0.0001, epochs=epochs, X=A0, Y=Y, batchSize=batchSize) 


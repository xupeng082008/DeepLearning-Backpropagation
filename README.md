# DeepLearning——Backpropagation
    A simple C++ implementation of Backpropagation which is designed to help beginners 
    to understand and study deep learning.


## Instruction
    It is a simple example of backpropagation,which includes 3 layers NeuralNetwork
    (Input_layer,Hidden_layer,Output_layer).There are 2 neurons in the Input_layer,
    2 neurons in the Hidden_layer,2 neurons in the Output_layer.I choose simgod function
    as the activation function and the function from is f(z) = 1 / (1 + exp(-z)),and the  
    function derivative is f(z)(1-f(z)) which is very helpful for the backpropagate 
    calculation.
    
    -main.cpp
        Give the NeuralNetwork two inputs, Initialize the network parameters (W,b).
    -class Neuron
    -class Layers
    -class NeuralNetwork

    You can train the NeuralNetwork with a appropriate error in main.cpp,and input what 
    you want ,the NeuralNetwork'output will give the result that you want ,just like a 
    AutoEncoder which is simple kind of deeplearning NeuralNetwork.


## Reference
    -http://deeplearning.stanford.edu/wiki/index.php/Backpropagation_Algorithm
    -https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
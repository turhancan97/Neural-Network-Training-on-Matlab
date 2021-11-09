# Laboratory of Basics of Smart System

**Author:** Turhan Can Kargın

**Software:** MATLAB

**This repository is my assignments on Basics of Smart Systems Laboratory Class which I took during my master degree at Poznan University of Technology. I share this with you to help you to basic understanding of artiicial neural network theory and application on MATLAB.**


![image](https://user-images.githubusercontent.com/22428774/140933705-de1f4079-b787-4413-b999-4f3a6a1631f2.png)

## Abstract

Artificial neural networks are becoming widely used in many areas such as complex robotics problems, computer vision, and classification problems. Working on artificial neural networks commonly referred to as “neural networks”, has been motivated right from its inception by the recognition that the human brain computes in an entirely different way from the conventional digital computer. The brain is highly complex, nonlinear and parallel computer. It has capability to organize its structural constituents, known as neurons, so as to perform certain computations (such as perception and motor control) very fast. Therefore, the interest of artificial neural network is to perform useful computations through a process of learning like our brain.
This report presents how artificial neural networks can be used in letter recognition and image recognition. The report is broadly categorized into two; the first part is a short overview on artificial neural networks, particularly its generalization property, as applied to systems identification. Second part contains some experimental design which shows how we can obtain neural network into applications.

## Introduction
### NEURAL NETWORK TOPOLOGIES AND RECALL:
Neural networks are closely modeled on biological processes for information processing, including specifically the nervous system and its basic unit, the neuron. Signals are propagated in the form of potential differences between the inside and outside of cells. The main components of a neuronal cell are shown in Figure 1. Dendrites bring signals from other neurons into the cell body or soma, possibly multiplying each incoming signal by a transfer weighting coefficient. In the soma, cell capacitance integrates the signals which collect in the axon hillock. Once the composite signal exceeds a cell threshold, a signal, the action potential, is transmitted through the axon. Cell nonlinearities make the composite action potential a nonlinear function of the combination of arriving signals. The axon connects through synapses with the dendrites of subsequent neurons. The synapses operate through the discharge of neurotransmitter chemicals across intercellular gaps, and can be either excitatory (tending to fire the next neuron) or inhibitory (tending to prevent firing of the next neuron) [1].

![image](https://user-images.githubusercontent.com/22428774/140934752-3058560e-7fce-4044-96eb-f51024154e09.png)
Figure 1 - Neuron Anatomy

### Neuron Mathematical Model:
Neural networks are set of algorithms inspired by the functioning of human brian. Generally when you open your eyes, what you see is called data and is processed by the Neurons (data processing cells) in your brain, and recognizes what is around you. That’s how similar the Neural Networks works. They takes a large set of data, process the data (draws out the patterns from data), and outputs what it is. What they do? Neural networks sometimes called as Artificial Neural networks (ANN’s), because they are not natural like neurons in your brain. They artificially mimic the nature and functioning of neural network. ANN’s are composed of a large number of highly interconnected processing elements (neurons) working in unison to solve specific problems. ANNs, like people, like child, they even learn by example. An ANN is configured for a specific application, such as pattern recognition or data classification, image recognition, voice recognition through a learning process. 
Neural networks, with their remarkable ability to derive meaning from complicated or imprecise data, can be used to extract patterns and detect trends that are too complex to be noticed by either humans or other computer techniques. A trained neural network can be thought of as an “expert” in the category of information it has been given to analyse. This expert can then be used to provide projections given new situations of interest and answer “what if” questions. Other advantages include: 
1.	Adaptive learning: An ability to learn how to do tasks based on the data given for training or initial experience. 
2.	Self-Organization: An ANN can create its own organization or representation of the information it receives during learning time. 

### Network layers:
The most common type of artificial neural network consists of three groups, or layers, of units: a layer of “input” units is connected to a layer of “hidden” units, which is connected to a layer of “output” units. 
* Input units - The activity of the input units represents the raw information that is fed into the network. This also called input layer. 
* Hidden units - The activity of each hidden unit is determined by the activities of the input units and the weights on the connections between the input and the hidden units. This also called hidden layer. 
* Output units - The behavior of the output units depends on the activity of the hidden units and the weights between the hidden and output units. this also called output layer.
 
![image](https://user-images.githubusercontent.com/22428774/140935526-bdb53ea8-0bc0-4fce-b80a-34a07afbe3a0.png)
Figure 2 - Neural Network with Input, Hidden and Output layer [2]

So the circles are called neurons. One list of neurons is a layer: the first layer is the input layer, while the last one (in our case, represented by only one neuron) is the output layer; those in the middle, on the other hand, are the hidden layers. Finally, all the interconnections among neurons are our synapses. 
The main elements of NN are, in conclusion, neurons and synapses, both in charge of computing mathematical operations. Yes, because NNs are nothing but a series of mathematical computations: each synapsis holds a weight, while each neuron computes a weighted sum using input data and synapses’ weights [3].

### Perceptrons:
Perceptrons — invented by Frank Rosenblatt in 1958, are the simplest neural network that consists of n number of inputs, only one neuron, and one output, where n is the number of features of our dataset. The process of passing the data through the neural network is known as forward propagation and the forward propagation carried out in a perceptron is explained in the following three steps. 

**Step 1:** For each input, multiply the input value xᵢ with weights wᵢ and sum all the multiplied values. Weights — represent the strength of the connection between neurons and decide how much influence the given input will have on the neuron’s output. If the weight w₁ has a higher value than the weight w₂, then the input x₁ will have a higher influence on the output than w₂.
![image](https://user-images.githubusercontent.com/22428774/140935896-92b05277-045c-41fe-8d27-dd56d67d9c61.png)

The row vectors of the inputs and weights are x = [x1, x2, … , xn] and w =[w1, w2, … , wn] respectively and their dot product is given by
 
![image](https://user-images.githubusercontent.com/22428774/140935938-acba93cd-b771-41bb-abaf-35113e7b9d30.png)

Hence, the summation is equal to the dot product of the vectors x and w

![image](https://user-images.githubusercontent.com/22428774/140935967-f7f29998-e34d-48b9-acb7-187cfc123294.png)

**Step 2:** Add bias b to the summation of multiplied values and let’s call this z. Bias — also known as the offset is necessary in most of the cases, to move the entire activation function to the left or right to generate the required output values.

![image](https://user-images.githubusercontent.com/22428774/140936012-727bdbda-21e1-4b67-97b6-5bc2cb9fe1fd.png)

**Step 3:** Pass the value of z to a non-linear activation function. Activation functions — are used to introduce non-linearity into the output of the neurons, without which the neural network will just be a linear function. Moreover, they have a significant impact on the learning speed of the neural network. Perceptrons have binary step function as their activation function. However, we shall use sigmoid — also known as logistic function as our activation function.

![image](https://user-images.githubusercontent.com/22428774/140936052-745968e4-ff41-48b1-abdb-44a4f4ca2441.png)

where σ denotes the sigmoid activation function and the output we get after the forward prorogation is known as the predicted value ŷ.

### Learning Algorithm:
The learning algorithm consists of two parts — backpropagation and optimization. 

#### Backpropagation: 
Backpropagation, short for backward propagation of errors, refers to the algorithm for computing the gradient of the loss function with respect to the weights. However, the term is often used to refer to the entire learning algorithm. The backpropagation carried out in a perceptron is explained in the following two steps. 

**Step 1:** To know an estimation of how far are we from our desired solution a loss function is used. Generally, mean squared error is chosen as the loss function for regression problems and cross entropy for classification problems. Let’s take a regression problem and its loss function be mean squared error, which squares the difference between actual (yᵢ) and predicted value (ŷᵢ ).

![image](https://user-images.githubusercontent.com/22428774/140936352-5235d466-d736-463c-8daf-70aeecb694c9.png)

Loss function is calculated for the entire training dataset and their average is called the Cost function C.

![image](https://user-images.githubusercontent.com/22428774/140936395-d12faa46-b63c-4dd1-85c3-6d384b39d92d.png)

**Step 2:** In order to find the best weights and bias for our Perceptron, we need to know how the cost function changes in relation to weights and bias. This is done with the help of the gradients (rate of change) — how one quantity changes in relation to another quantity. In our case, we need to find the gradient of the cost function with respect to the weights and bias. Let’s calculate the gradient of cost function C with respect to the weight wᵢ using partial derivation. Since the cost function is not directly related to the weight wᵢ, let’s use the chain rule.

![image](https://user-images.githubusercontent.com/22428774/140936444-3a68ea91-d27c-4a93-b2b5-788b705aef7d.png)

Now we need to find the following three gradients

![image](https://user-images.githubusercontent.com/22428774/140936466-ba80e24a-9300-4b6f-99ef-42ea27425de1.png)

Let’s start with the gradient of the cost function (C) with respect to the predicted value ( ŷ )

![image](https://user-images.githubusercontent.com/22428774/140936485-13c8d88d-a530-4e3b-af40-bf404616a4a5.png)

Let y = [y1 , y2 , … yn] and ŷ =[ ŷ1 , ŷ2 , … ŷn] be the row vectors of actual and predicted values. Hence the above equation is simplified as

![image](https://user-images.githubusercontent.com/22428774/140936529-68a293a1-5378-4d90-8c57-ec5a93484b05.png)

Now let’s find the gradient of the predicted value with respect to the z. This will be a bit lengthy.

![image](https://user-images.githubusercontent.com/22428774/140936560-0dd8fa34-f2c8-4bb4-8d45-a3da684291eb.png)

The gradient of z with respect to the weight wᵢ is

![image](https://user-images.githubusercontent.com/22428774/140936616-8785d818-9421-4687-9199-09f32887d6a2.png)

Therefore we get,

![image](https://user-images.githubusercontent.com/22428774/140936638-52ba299c-b781-443d-8698-e09d0f1fbc2e.png)

What about Bias? — Bias is theoretically considered to have an input of constant value 1. Hence,

![image](https://user-images.githubusercontent.com/22428774/140936669-4be9096a-99cd-45cc-8d35-65e58b9260c8.png)

#### Optimization:
Optimization is the selection of the best element from some set of available alternatives, which in our case, is the selection of best weights and bias of the perceptron. Let’s choose gradient descent as our optimization algorithm, which changes the weights and bias, proportional to the negative of the gradient of the cost function with respect to the corresponding weight or bias. Learning rate (α) is a hyperparameter which is used to control how much the weights and bias are changed.
The weights and bias are updated as follows and the backpropagation and gradient descent is repeated until convergence [4].

![image](https://user-images.githubusercontent.com/22428774/140936931-ffc3c456-4017-4707-919f-4ec23d34b771.png)

I have explained the working of a single neuron in the introduction part of this report. However, these basic concepts are applicable to all kinds of neural networks with some modifications. Now, let’s tell our materials for the projects and start to explain the experimental design. I will talk about first how to create a neural network in MATLAB and then I will explain three different projects which are letter recognition and image recognition with Neural Network in the experimental design part of the report.


  


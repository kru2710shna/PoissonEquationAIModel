# PoissonEquationAIModel
---


## Problem Defination

Let's consider a simple boundry value problem of Poisson equation in 1D space. 

! [](Images/Screenshot 2025-07-16 at 10.13.35 PM.png)

Boundry Conditon
y(-1) = 0 and y(1) = 0

The Analytical Solution for this problem is y(x) = sin(pie.x)

Universal Approximation Theorem
The Universal Approximation Theorem states that a simple neural network with at least one hidden layer and finite number of neurons, can approximate any continuous function on a closed and bounded interval to any desired level of accuracy, given enough neurons in the hidden layer and appropriate activation functions.

Simply put, The Universal Approximation Theorem states that neural networks can approximate any function. This means that no matter what the function f(x) is, there exists a neural network that can get very close to the correct output. This applies to any number of inputs and outputs.


Neural Network Setup
We design a simple Neural Network to approximate the solution y(x). The input to the network will be the spatial coordinate x, and the output will be the value of y at x.
where, yvy is the neural network.


The Concept of Physics-Based Machine Learning
Traditional Machine Learning vs. PINNs
In traditional machine learning, models are trained using datasets consisting of input-output pairs. For example, in supervised learning, a dataset might consist of input features X and corresponding output features Y. The model learns to map X to Y by minizing the error between its predictions and the true output featues.
In contrast, PINNs do not rely on a traditional dataset. Instead, they are trained using the governing physical laws of the system, typically expressed in the form of differential equations. The "data" in this context comes from the physics itself, which includes:
- The governing equations
1. Governing Differential Equations: These equations describe the behavior of the system.
2. Boundary and Initial Conditions: These conditions specify the state of the system at the boundaries and initial time.


Detailed Flow Chart

1. Define Neural Network
• Inpiit: Spatial coordinate x.
• Hidden Layers: Non-linear transformations to approximate the solution.
• Output: Predicted yNN(x).

2. Compute Derivatives
• Use tf.GradientTape to compute By.

3. Define Residual
• Compute the residual of the differential equation

4. Loss Function
• Combine residual loss and boundary loss
• Residual loss: Lres = 1/2 * ∫[−1,1]
loss = MSE(residual) + MSE(boundary conditions)

5. Training
• Minimize the loss function using an optimizer (e.g., Adam).

6. Evaluation
• Predict and plot the output y over the domain and compare with the analytical solution.
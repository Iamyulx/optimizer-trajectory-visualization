# Optimizer Trajectory Visualization 
This project visualizes how different optimization algorithms move across a loss landscape during training.
The experiment compares the trajectories of:

SGD

Adam

AdamW

on the classic Rosenbrock function, a common benchmark for optimization algorithms.

The goal is to build intuition about how optimizers navigate curved loss surfaces.

# Rosenbrock Function

The Rosenbrock function is a well-known optimization test problem defined as:

f(x,y) = (1 - x)^2 + 100(y - x^2)^2

It is challenging because:

The global minimum lies inside a narrow curved valley

Many optimization algorithms struggle to follow this valley efficiently

Global minimum:

(x, y) = (1, 1)

with

f(x,y) = 0


# Optimizers Implemented

The project implements three optimizers from scratch.

## SGD

Stochastic Gradient Descent updates parameters using the gradient direction:

x = x − lr * grad

SGD is simple but can struggle with curved valleys and poorly conditioned problems.

## Adam

Adam (Adaptive Moment Estimation) uses:

momentum (first moment)

adaptive learning rates (second moment)

Update rule:

m_t = β1 m_{t-1} + (1 − β1) g_t
v_t = β2 v_{t-1} + (1 − β2) g_t²

with bias correction applied before the update.

## AdamW

AdamW improves Adam by decoupling weight decay from gradient updates.

Instead of adding L2 regularization to the gradient, weight decay is applied directly to the parameters.

θ ← θ − lr * update
θ ← θ − lr * λ * θ

AdamW is widely used in modern architectures such as:

Transformers

Vision Transformers

Large Language Models

# Experiment

The experiment:

Initializes parameters at

x = (-2, 2)

Runs each optimizer for 200 steps

Records the trajectory of parameters

Plots the trajectories over the loss landscape

# Result

The contour plot shows how each optimizer moves through the loss surface.

Observations:

SGD moves slowly and struggles with the curved valley

Adam converges faster due to adaptive learning rates

AdamW behaves similarly to Adam but includes decoupled weight decay

# Run the Project

Clone the repository:

git clone https://github.com/Iamyulx/optimizer-trajectory-visualization
cd optimizer-trajectory-visualization

Install dependencies:

pip install -r requirements.txt

Run the visualization:

python experiments/rosenbrock_visualization.py


# Concepts Demonstrated

This project demonstrates understanding of:

optimization algorithms

gradient-based learning

loss landscapes

adaptive optimizers

weight decay regularization

optimization dynamics

# Why This Project Matters

Understanding optimizer behavior is critical for training deep neural networks.

Visualization helps build intuition about:

convergence speed

optimization stability

behavior in non-convex loss landscapes

These concepts are fundamental in modern machine learning systems.



optimizer-trajectory-visualization
│
├── README.md
├── requirements.txt
│
├── optimizers
│   ├── sgd.py
│   ├── adam.py
│   └── adamw.py
│
├── experiments
│   └── rosenbrock_visualization.py
│
└── figures
    └── optimizer_trajectories.png

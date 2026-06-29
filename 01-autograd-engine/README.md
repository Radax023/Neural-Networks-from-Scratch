# 01 - Autograd Engine (Micrograd)

This folder contains the core implementation of a scalar-valued autograd engine and a mini-neural network library constructed from scratch.

### Key Concepts Implemented:
- **Graph Execution & Backpropagation:** Building computational graphs dynamically and tracking derivatives using the chain rule.
- **Topological Sort:** Implementing ordering algorithms to ensure backward passes execute only when all dependent gradients are resolved.
- **Exercises Handled:** Gradient derivation, numerical gradient checking, softmax/NLL implementation, and PyTorch gradient verification.

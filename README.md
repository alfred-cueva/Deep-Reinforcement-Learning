# Deep Reinforcement Learning (Fall 2025)

A comprehensive collection of graduate-level assignments covering modern Deep Reinforcement Learning algorithms and techniques.

## 📋 Course Overview

This repository contains four homework assignments that progressively build from foundational RL concepts to state-of-the-art methods. The course covers both **model-free** and **model-based** reinforcement learning, as well as **inverse RL** and modern LLM fine-tuning techniques.

## 📚 Assignments

### HW1: Policy Gradients & Proximal Policy Optimization
**Topics:** REINFORCE, Vanilla Policy Gradients, PPO (Proximal Policy Optimization)
- Covers the gradient of the objective function with respect to policy parameters
- Implements foundational policy gradient algorithms
- Progresses to modern PPO implementation
- 📄 [DRL_HW1.ipynb](DRL_HW1.ipynb) | [DRL_HW1_AlfredCueva.pdf](DRL_HW1_AlfredCueva.pdf)

### HW2: Q-Learning & Actor-Critic Methods
**Topics:** DQN, DDPG, Soft Actor-Critic (SAC)
- Builds on tabular Q-learning
- Covers Deep Q Networks (DQN)
- Deep Deterministic Policy Gradients (DDPG)
- Soft Actor-Critic implementation (SAC)
- 📄 [DRL_HW2.ipynb](DRL_HW2.ipynb) | [DRL_HW2_AlfredCueva.pdf](DRL_HW2_AlfredCueva.pdf)

### HW3: Model-Based Reinforcement Learning
**Topics:** Neural Dynamics Modeling, Cross Entropy Method (CEM), PETS
- Explores model-based RL vs model-free approaches
- Deterministic neural dynamics modeling
- Cross Entropy Method
- Stochastic neural dynamics modeling
- Probabilistic Ensembles with Trajectory Sampling (PETS)
- 📄 [DRL_HW3.ipynb](DRL_HW3.ipynb) | [DRL_HW3_AlfredCueva.pdf](DRL_HW3_AlfredCueva.pdf)

### HW4: Inverse RL & LLM Fine-Tuning
**Topics:** Inverse Reinforcement Learning, GRPO, QLoRA
- Maximum Entropy Inverse RL (MaxEnt IRL)
- Reward modeling from expert demonstrations
- Group Relative Policy Optimization (GRPO)
- QLoRA-adapted large language model fine-tuning
- Structured reasoning format training for LLMs
- 📄 [DRL_HW4.ipynb](DRL_HW4.ipynb) | [DRL_HW4_AlfredCueva.pdf](DRL_HW4_AlfredCueva.pdf)

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- NumPy & SciPy
- OpenAI Gym or similar environments
- Google Colab recommended (especially for GPU access in HW4)

## 📖 How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alfred-cueva/Deep-Reinforcement-Learning.git
   cd Deep-Reinforcement-Learning
   ```

2. **View assignments:** Open any `.ipynb` file in Jupyter Notebook or Google Colab

3. **Review solutions:** PDF versions are available for each assignment

## 🔗 Key Concepts Progression

- **HW1:** How to optimize policies directly via gradients
- **HW2:** How to learn value functions in continuous spaces
- **HW3:** How to learn environment dynamics and plan with them
- **HW4:** How to learn rewards from data and fine-tune LLMs

## 📝 Notes

- All notebooks include warm-up questions, theoretical explanations, and implementation sections
- Google Colab is recommended for optimal execution
- HW4 requires GPU access for GRPO fine-tuning sections
- PDF solutions include written answers and implementation results

---
*Graduate Course - Deep Reinforcement Learning, Fall 2025*

# GolfRL 🏌️‍♂️🤖

This project explores **robotic manipulation with reinforcement learning (RL)** by training a **Franka Emika Panda robot arm** to play golf in simulation.  
The task is to learn a policy that enables the robot to grasp the golf club, align it with the ball, and strike towards a goal.  

The project is built using:  
- [Stable-Baselines3 (SB3)](https://github.com/DLR-RM/stable-baselines3) for RL algorithms  
- [Gymnasium](https://gymnasium.farama.org/) for standardized RL interfaces  
- [SAI](https://competesai.com/) (Simulation AI) platform for environment hosting  
- [MuJoCo](https://mujoco.org/) as the physics engine  

---

## 🎥 Demo


![Demo](videos/training_model.gif)


---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vasDEV2/GolfRL.git
cd GolfRL


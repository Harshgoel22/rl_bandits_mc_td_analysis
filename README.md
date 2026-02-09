# Reinforcement Learning: Bandits & Value Prediction (MC vs TD)

**Course:** CS780 — Deep Reinforcement Learning

**Course Instructor:** Prof. Ashutosh Modi

**Assignment:** Assignment 1

---

## 📌 Project Overview

This project presents an empirical study of **Exploration–Exploitation strategies in Multi-Armed Bandits** and **Value Prediction using Monte-Carlo and Temporal Difference learning** using **Gym-style environments**.

All implementations, experiments, analysis, and visualizations are contained in a **single Jupyter Notebook**.

---

## 🔹 Part 1 — Multi-Armed Bandits

Experiments are conducted on **2-armed** and **10-armed** bandit environments.

### Algorithms Implemented

* Pure Exploration
* Greedy (Pure Exploitation)
* ε-Greedy
* Decaying ε-Greedy
* Softmax (Boltzmann Exploration)
* Upper Confidence Bound (UCB)

### Performance Metrics

* Average Reward vs Episodes
* Optimal Action Percentage vs Episodes
* Cumulative Regret vs Episodes

---

## 🔹 Part 2 — Value Prediction (Random Walk)

Prediction algorithms are implemented on the **7-state Random Walk Environment**.

### Algorithms

* First-Visit Monte Carlo (FVMC)
* Every-Visit Monte Carlo (EVMC)
* Temporal Difference Learning (TD(0))

### Analysis & Visualizations

* Value Estimates vs Episodes
* Smoothed Value Estimate Curves (Variance Reduction via Multiple Environments)
* Target Value (Return) Curves
* Comparison with True State Values

To obtain smoother and more stable learning curves, value estimates were averaged across multiple independent environment runs, which reduces variance and improves convergence interpretation.

---

## 📊 Generated Plots

### Bandit Analysis

* Average Reward Curve
* Optimal Action % Curve
* Regret Curve

### Prediction Analysis

* Value Estimates vs Episodes
* Smoothed Convergence Curves (FVMC, EVMC, TD)
* Target Value Curves

---

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install numpy matplotlib gym jupyter
```

### 2. Launch Notebook

```bash
jupyter notebook
```

Open the notebook:

```
RL_Bandits_MC_TD_Analysis.ipynb
```

Run all cells sequentially to reproduce experiments and plots.

---

## 🧪 Key Observations

* UCB demonstrates strong performance due to optimism-driven exploration.

* Decaying ε-Greedy performs better than fixed ε in long-term reward.

* Softmax performance is sensitive to temperature scheduling.

* In this experimental setup, **First-Visit Monte Carlo (FVMC) produced the most accurate value estimates**, outperforming EVMC and TD in terms of final convergence to true values. FVMC is **unbiased**, but has higher variance.

* **Temporal Difference (TD) learning shows better performance when the number of episodes is very large.** Since TD uses bootstrapping, it produces **biased but low-variance estimates**, which makes learning more stable over long runs.

* Because TD is biased, individual updates may be affected by noisy or outlier returns. Increasing the number of episodes helps reduce the impact of such noise, allowing TD to converge effectively.

* Monte-Carlo methods are unbiased but exhibit higher variance, while TD provides lower variance at the cost of slight bias — illustrating the classic **bias–variance tradeoff** in reinforcement learning.

* Averaging across multiple environments significantly reduces variance, producing smoother and more interpretable learning curves.

---

## 🧠 Concepts Covered

* Exploration vs Exploitation
* Regret Minimization in Multi-Armed Bandits
* Monte-Carlo vs Temporal Difference Learning
* Bias–Variance Tradeoff in Reinforcement Learning
* Convergence Behavior of Prediction Algorithms
* Effect of Temperature (Softmax) and Confidence Bound (UCB)
* Variance Reduction via Multi-Environment Averaging

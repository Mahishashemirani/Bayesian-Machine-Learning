# 🌀 Levy Stable Models of Stochastic Volatility

## 📈 Introduction

The Levy Stable Models of Stochastic Volatility extend classical models by allowing for heavy-tailed distributions and jumps. This approach provides a more flexible framework for modeling asset prices and capturing the inherent volatility observed in financial markets. 

In this project, we implemented Levy stable models using a toy dataset, incorporating methods to estimate the stability and scale parameters.

---

## 📊 Differences from Classical Models

1. **Heavy-Tailed Behavior**:
   - Classical models often assume normal distributions, which can underestimate extreme market movements.
   - Levy stable models can capture extreme events better due to their heavy-tailed nature.

2. **Parameter Estimation**:
   - Traditional methods provide point estimates for parameters.
   - Levy models incorporate uncertainty in parameter estimation through Bayesian methods.

3. **Flexibility**:
   - Levy stable distributions can model a wide variety of behaviors in financial returns, making them versatile in practice.

---

## 🏗️ Model Specification

Given a 2D input feature matrix \(X\) and an output vector \(y\), the Levy Stable Model is defined as:

\[
y = X \cdot w + b + \epsilon
\]

- **\(w\)**: Weight vector with a **Normal prior** \(w \sim \mathcal{N}(0, I)\)  
- **\(b\)**: Bias term with a **Normal prior** \(b \sim \mathcal{N}(0, 1)\)
- **\(\epsilon\)**: Noise term with a **Levy stable prior**
- **\(y|x\)**: likelihood term with a probability distribution \(y \sim \text{LevyStable}(X \cdot w + b, \text{scale})\)

This model captures uncertainty by placing priors on the weights and bias. After observing data, the posterior distribution is updated to reflect the new information.

---

## 🔍 Inference using Stochastic Variational Inference (SVI)

In Bayesian models, exact inference is often intractable, especially for high-dimensional problems. Therefore, we use **Stochastic Variational Inference (SVI)**, which approximates the posterior distribution by minimizing the **Kullback-Leibler (KL) divergence** between the true posterior and a variational approximation.

### Key Inference Patterns with SVI

1. **Learning Variational Parameters**:  
   Instead of directly learning the parameters \(w\) and \(b\), SVI learns the **mean and variance of their variational distributions** (e.g., Normal).

2. **Uncertainty Propagation**:  
   As training progresses, the model learns both the **mean** and **uncertainty** of each parameter. Predictions also reflect this uncertainty by sampling from the learned distributions.

3. **Trade-off between Accuracy and Uncertainty**:  
   In SVI, the optimization involves balancing **data fit** (likelihood) and **model complexity** (prior regularization). As a result, the learned posteriors incorporate both the observed data and prior beliefs.

4. **Convergence Patterns**:  
   During training, the **ELBO (Evidence Lower Bound)** serves as the objective function to be maximized. A **steady increase in ELBO** indicates the model is learning an optimal approximation to the true posterior.

---

## ⚙️ Training and Loss Behavior

Training a **Levy Stable Model** involves optimizing the **Evidence Lower Bound (ELBO)** to approximate the posterior distribution of the parameters. Unlike classical regression, which aims to minimize a straightforward objective (e.g., Mean Squared Error), Levy models balance **data fit** and **regularization from the prior distributions**. As a result, the loss function reflects not only how well the model fits the data but also how it adjusts parameter uncertainty.

During training, the **loss function tends to fluctuate more** compared to classical models because:

1. **Posterior Sampling**: At each step, the model samples from variational distributions, adding randomness to the optimization.
2. **KL Divergence Optimization**: The KL term in the ELBO makes optimization more complex, leading to occasional jumps in the loss.
3. **Exploration vs. Exploitation Trade-off**: The model tries to strike a balance between exploring uncertain parameter regions and exploiting regions with better fit to the data.

These fluctuations are natural and expected in **variational inference** processes. As training progresses, the model typically converges, but the path to convergence can exhibit significant **noise** compared to the smooth curve seen in classical models.

### 📉 Loss Over Training

The following plot shows the loss function over the training epochs, illustrating the fluctuations characteristic of the optimization process:

![Loss Function Plot](Plots/training%20loss.png)

---

## 📊 Results

### 📈 Dataset Plot

Below is the plot of the dataset used in the regression task:

![Price Plot](Plots/price.png)

### 🔍 Additional Results

- **Daily Log Returns**:
  
  ![Daily Log Return](Plots/daily%20log%20return.png)

- **Empirical Distribution of Returns**:
  
  ![Empirical Distribution](Plots/empirical%20distribution.png)

- **Training Loss**:
  
  ![Training Loss](Plots/loss%20over%20training.png)

- **Posterior Predictive Check**:
  
  ![Posterior Predictive](Plots/posterior%20predictive.png)

- **Predicted Histogram of Returns**:
  
  ![Predicted Histogram](Plots/predicted%20histogram.png)

---

### 📊 Parameter Estimation

The following table compares the **true parameter values** with the **estimated values** from the Levy stable model. The estimates reflect the **posterior mean** of the parameters, while the **standard deviation** serves as the **uncertainty (confidence interval)** for each parameter.

| Parameter             | True Value | Estimated Value | Standard Deviation (±) |
|-----------------------|------------|-----------------|------------------------|
| Stability Parameter \(s\)  |  1.5       | 1.482           | 0.084                  |
| Scale Parameter \(β\)      |  0.5       | 0.498           | 0.041                  |

---

### 🌟 Interpretation

- **Posterior Mean**: Represents the most likely value for each parameter based on the observed data and prior knowledge.
- **Standard Deviation**: Provides insight into the **uncertainty** of each parameter, with larger values indicating greater uncertainty.

This table highlights the model's ability to accurately estimate parameters while incorporating **uncertainty quantification** through Bayesian inference.

---


### 🌟 Interpretation of Posterior Distributions

The above plots demonstrate how the posterior distributions start with **high uncertainty** (wide spread) and gradually become more focused around the **true parameter values** as more iterations are performed. This process reflects how **Bayesian inference** learns from data while incorporating **uncertainty** throughout the optimization.

As seen in the **standard deviations** from the table, the final parameter estimates capture both the **mean value** and the **uncertainty range**, which is essential for robust predictions in Levy stable models.

---

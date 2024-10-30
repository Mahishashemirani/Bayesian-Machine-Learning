# 🌀  Volatility Modelling using Alpha Stable distribution
# 📈Introduction

## Aim of the Analysis

The objective of this analysis is to model and compare the volatility of Bitcoin and Gold prices using a **Bayesian model** with an **alpha-stable distribution**. The core idea is to assess which of these assets is more prone to **rare events and heavy-tailed behavior**. 

By examining the **tail properties** of the fitted alpha-stable distributions, we aim to determine whether Bitcoin or Gold exhibits a higher risk of extreme price movements. This comparison will help us understand the stability and risk characteristics of both assets, which is crucial for portfolio management, hedging strategies, and risk assessment.

---

## What is an Alpha-Stable Distribution?

The **alpha-stable distribution** generalizes the Gaussian distribution, allowing for **skewness** and **heavy tails**. It is widely used in finance to model returns that deviate from the assumptions of normality (such as **fat tails** and **asymmetric behavior**). 

Unlike a normal distribution, the alpha-stable distribution can capture **extreme events** with higher probability, which makes it suitable for modeling financial time series prone to **black swan events**.

The key parameters of the alpha-stable distribution are:
- **α (alpha)**: Stability parameter (0 < α ≤ 2). Lower values indicate **heavier tails** (higher chance of rare events).
- **β (beta)**: Skewness parameter (−1 ≤ β ≤ 1). Determines the asymmetry of the distribution.
- **γ (gamma)**: Scale parameter. Controls the spread or variability.
- **δ (delta)**: Location parameter. Represents the central tendency or shift.

For normal distributions, **α = 2**. As **α decreases**, the distribution becomes more "heavy-tailed," indicating a higher likelihood of rare, extreme movements.


## 📊 Differences from Classical Models

1. **Heavy-Tailed Behavior**:
   - Classical models often assume normal distributions, which can underestimate extreme market movements.
   - Levy stable models can capture extreme events better due to their heavy-tailed nature.

2. **Parameter Estimation**:
   - Traditional methods provide point estimates for parameters.
   - Bayesian models incorporate uncertainty in parameter estimation through Bayesian methods.

## Log-Returns of Bitcoin and Gold

The following plot provides an overview of the **log-returns** for Bitcoin and Gold over the given time period. Log-returns are useful for analyzing financial time series because they help normalize price changes, making the data more comparable over time. Observing the spikes in the log-returns can indicate times of extreme events or sudden market movements.

### Plot of Bitcoin and Gold Log-Returns

![Bitcoin and Gold Log-Returns](Plots/Log_returns.png)

---

### Interpretation

- **Bitcoin**: As observed, Bitcoin shows higher volatility with several significant spikes, indicating frequent large price movements.
- **Gold**: Gold appears more stable, with fewer extreme movements compared to Bitcoin. However, rare events are still present in the form of small but sharp deviations.
---
## Empirical Distributions of Bitcoin and Gold Log-Returns

In addition to the time-series plot of log-returns, the following empirical distribution plots provide insights into the **probability distribution** of the log-returns for Bitcoin and Gold. 

### Why Empirical Distributions Matter?
The shape of the empirical distribution helps us understand the **heavy-tailed behavior** of each asset. A **normal distribution** assumption in financial models would underestimate the probability of extreme events, but the empirical distributions of Bitcoin and Gold often show **fatter tails**. This makes them suitable for modeling with **alpha-stable distributions**.

### Bitcoin Empirical Distribution
![Bitcoin Empirical Distribution](Plots/Bitcoin_Empirical.png)

---

### Gold Empirical Distribution
![Gold Empirical Distribution](Plots/Gold_Empirical.png)

---

### Comparison and Relevance
By visually comparing the two distributions:
- **Bitcoin** shows significantly fatter tails, reinforcing the need for non-Gaussian models like the alpha-stable distribution.
- **Gold** has thinner tails, meaning rare events are less likely, which aligns with its role as a stable asset in the financial markets.

These plots support the quantitative comparison of the **stability parameters (α)** in the next section, where we fit alpha-stable distributions to the log-returns of both assets.


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

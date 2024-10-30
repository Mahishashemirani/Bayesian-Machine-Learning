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
![Bitcoin Empirical Distribution](Plots/Bitcoin_Emperical.png)

---

### Gold Empirical Distribution
![Gold Empirical Distribution](Plots/Gold_Emperical.png)

---

### Comparison and Relevance
By visually comparing the two distributions:
- **Bitcoin** shows significantly fatter tails, reinforcing the need for non-Gaussian models like the alpha-stable distribution.
- **Gold** has thinner tails, meaning rare events are less likely, which aligns with its role as a stable asset in the financial markets.

These plots support the quantitative comparison of the **stability parameters (α)** in the next section, where we fit alpha-stable distributions to the log-returns of both assets.
## Mathematical Framework

In this analysis, the **log-returns** of Bitcoin and Gold prices are modeled using the following equations:


$\log p_t = \log p_{t-1} + \sqrt{h_t}$  

$\log h_t = \log h_{t-1} + \sigma \cdot v_t$

### Explanation of the Equations

1. **First Equation**:  
   This equation models the **log-price dynamics**.  
   - $\(\log p_t\)$ is the log-price at time \(t\).  
   - $\(\log p_{t-1}\)$ is the log-price at time \(t-1\).  
   - $\(\sqrt{h_t}\)$ accounts for **stochastic volatility**. This introduces randomness, meaning price changes are influenced by the **current level of volatility**.

2. **Second Equation**:  
   The second equation models the **evolution of volatility over time**, capturing **volatility clustering** (i.e., periods of high volatility followed by more high volatility).  
   - $\(\log h_t\)$ is the log-volatility at time \(t\).  
   - $\(\log h_{t-1}\)$ is the log-volatility at time \(t-1\).  
   - $\(v_t\)$ is a **stochastic term** representing the shocks to volatility.  
   - $\(\sigma\)$ is a **parameter** controlling the sensitivity of volatility to these shocks.

### Key Idea: Volatility Clustering

The **time-dependent volatility** (second equation) allows us to model **volatility clustering**, which is common in financial markets. This means that large price movements are often followed by large movements, and small movements are followed by small ones. 

---

## Implementation

Our model assumes that the **log-returns** of Bitcoin and Gold are distributed as:

$\log R_t \sim \text{Stable}(\alpha, \beta, \sqrt{h_t}, r_{\text{loc}})$

where:
- $\(\alpha\)$ is the **stability** parameter.
- $\(\beta\)$ is the **skewness** parameter.
- $\(\sqrt{h_t}\)$ is the **scale** or volatility parameter.
- $\(r_{\text{loc}}\)$ is the **location** parameter.

The goal of our analysis is to **infer these parameters** from the log-returns using **Stochastic Variational Inference (SVI)** to maximize the **Evidence Lower Bound (ELBO)**. Below is the implementation summary along with the necessary reparameterization details.

---

### SVI and ELBO Optimization

In Bayesian inference, **maximizing the ELBO** allows us to estimate the posterior distribution of the parameters efficiently. SVI is chosen for this task because it scales well with large datasets like financial time series.

The following plot shows the convergence of the ELBO during training, indicating that the model successfully learned the parameters:

![ELBO Convergence Plot](Plots/SVI_loss.png)

As seen in the plot, the ELBO increases and stabilizes over time, suggesting that the variational inference is working as expected.

---

### Reparameterization of the Stable Distribution

Inference for **non-symmetric stable distributions** can be challenging because the **log-probability function (log_prob)** is not implemented for the general case. To overcome this, we use a **reparameterization trick** provided by Pyro that reparameterizes a Stable random variable as the sum of two other stable random variables:  
1. One symmetric  
2. One totally skewed  
Reparameterizing the stable distribution makes the model more computationally efficient during optimization, reducing the complexity of sampling and inference.

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

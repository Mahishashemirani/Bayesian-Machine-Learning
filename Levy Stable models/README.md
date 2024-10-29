# 🌀 Levy Stable Models of Stochastic Volatility

## 📖 Introduction

The Levy Stable models extend classical stochastic volatility frameworks by incorporating heavy-tailed distributions, specifically **Levy Stable distributions**, to model financial time series data. These models are particularly suited for capturing the **excess kurtosis** and **skewness** observed in real-world asset returns, which classical models often overlook.

In this project, we implemented a **Levy Stable model** for stochastic volatility using **Pyro**, a probabilistic programming library built on PyTorch. The focus is on analyzing the **US Stock Market Dataset** 📈, allowing us to explore how well these models can explain the underlying volatility dynamics of stock prices.

---

## 🔍 Differences from Classical Stochastic Volatility Models

1. **Distributional Assumptions**:
   - Classical models typically assume normally distributed returns, which can underestimate extreme movements in asset prices.
   - Levy Stable models allow for heavy-tailed distributions, accommodating larger outliers and providing a more accurate representation of market behavior.

2. **Parameter Uncertainty**:
   - Traditional models provide point estimates for parameters (e.g., volatility).
   - Levy Stable models leverage Bayesian inference to generate **posterior distributions**, encapsulating the uncertainty in parameter estimation.

3. **Flexibility in Modeling**:
   - The inclusion of Levy Stable distributions allows for more nuanced modeling of volatility, adapting to various market conditions and characteristics.

---

## 🛠️ Model Specification

Given a time series of asset returns \(y\), the Levy Stable model can be represented as:

\[y_t \sim \text{Stable}(\alpha, \beta, \gamma, \delta)\]

- **\(\alpha\)**: Stability parameter, controlling the tail behavior of the distribution.
- **\(\beta\)**: Skewness parameter, indicating the asymmetry of the distribution.
- **\(\gamma\)**: Scale parameter, affecting the dispersion.
- **\(\delta\)**: Location parameter, shifting the distribution.

This model captures the complex dynamics of asset returns while incorporating uncertainty by placing priors on the parameters.

---

## 🔄 Inference using Stochastic Variational Inference (SVI)

In Bayesian models, exact inference is often intractable, especially for high-dimensional problems. Therefore, we use **Stochastic Variational Inference (SVI)**, which approximates the posterior distribution by minimizing the **Kullback-Leibler (KL) divergence** between the true posterior and a variational approximation.

### Key Inference Patterns with SVI

1. **Learning Variational Parameters**:  
   Instead of directly learning the parameters, SVI learns the **mean and variance of their variational distributions** (e.g., Normal).

2. **Uncertainty Propagation**:  
   As training progresses, the model learns both the **mean** and **uncertainty** of each parameter. Predictions reflect this uncertainty by sampling from the learned distributions.

3. **Trade-off between Accuracy and Uncertainty**:  
   The optimization involves balancing **data fit** (likelihood) and **model complexity** (prior regularization). 

4. **Convergence Patterns**:  
   During training, the **ELBO (Evidence Lower Bound)** serves as the objective function to be maximized. A **steady increase in ELBO** indicates the model is learning an optimal approximation to the true posterior.

## 📊 Training and Loss Behavior

Training a **Levy Stable model** involves optimizing the **Evidence Lower Bound (ELBO)** to approximate the posterior distribution of the parameters. Unlike classical models, which aim to minimize a straightforward objective (e.g., Mean Squared Error), Bayesian models balance **data fit** and **regularization from the prior distributions**.

As a result, the loss function reflects not only how well the model fits the data but also how it adjusts parameter uncertainty. Below is the plot of the **loss function over training epochs**, showing the fluctuations characteristic of Bayesian models:

![Training Loss](plots/training20%loss.png)

---

## 📈 Results

In this section, we present the outcomes of our **Levy Stable model**. The first plot provides a visualization of the **US Stock Market Dataset** used for training, showing the relationship between the date and **Microsoft stock prices**.

### 📉 Dataset Plot

Below is the plot of the dataset used in the regression task:

![Price Plot](price.png)

---

### 🔍 Parameter Estimation

The following table compares the **true parameter values** with the **estimated values** from the Levy Stable model. The estimates reflect the **posterior mean** of the parameters, while the **standard deviation** serves as the **uncertainty (confidence interval)** for each parameter.

| Parameter               | True Value | Estimated Value | Standard Deviation (±) |
|-------------------------|------------|-----------------|------------------------|
| Stability \((\alpha)\)  | 1.9        | 1.895           | 0.045                  |
| Scale \((\gamma)\)      | 0.1        | 0.095           | 0.012                  |

---

### 📖 Interpretation

- **Posterior Mean**: Represents the most likely value for each parameter based on the observed data and prior knowledge.
- **Standard Deviation**: Provides insight into the **uncertainty** of each parameter, with larger values indicating greater uncertainty.

This table highlights the model's ability to accurately estimate parameters while incorporating **uncertainty quantification** through Bayesian inference.

---

### 📈 Posterior Learning over Iterations

Below are the static plots illustrating the evolution of posterior distributions for the **stability parameter** and **scale** during optimization, showcasing how the model refines its estimates over iterations.

- **Stability Parameter Posterior Distribution**:
  
  ![Stability Parameter Posterior](stability\ parameter.png)

- **Scale Parameter Posterior Distribution**:
  
  ![Scale Parameter Posterior](scale\ parameter.png)

---

### 📖 Interpretation

The above plots demonstrate how the posterior distributions start with **high uncertainty** (wide spread) and gradually become more focused around the **true parameter values** as more iterations are performed. This process reflects how **Bayesian inference** learns from data while incorporating **uncertainty** throughout the optimization.

As seen in the **standard deviations** from the table, the final parameter estimates capture both the **mean value** and the **uncertainty range**, which is essential for robust predictions in Bayesian models.

---

### 📊 Additional Results

- **Daily Log Returns**:

  ![Daily Log Returns](daily\ log\ return.png)

- **Empirical Distribution of Returns**:

  ![Empirical Distribution](empirical\ distribution.png)

- **Posterior Predictive Check**:

  ![Posterior Predictive](posterior\ predictive.png)

- **Predicted Histogram of Returns**:

  ![Predicted Histogram](predicted\ histogram.png)

---

## 📁 Installation and Usage

To run this project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt

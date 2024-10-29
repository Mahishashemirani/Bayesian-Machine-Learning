# Bayesian Logistic Regression with Pyro

## 📘 Introduction

Bayesian Logistic Regression extends classical logistic regression by treating model parameters as **random variables** instead of fixed values. This approach incorporates **uncertainty** in model predictions, providing a **distribution over parameters and predictions**. Bayesian inference estimates **posterior distributions** of parameters, based on prior beliefs and observed data.

This project implements **Bayesian Logistic Regression using Pyro** on a toy dataset, with **Normal priors** placed over the regression parameters (weights and bias).

---

## ⚖️ Differences from Classical Logistic Regression

1. **Parameter Estimation**:
   - Classical logistic regression uses point estimates (e.g., Maximum Likelihood Estimation) for parameters.
   - Bayesian logistic regression defines parameters as **random variables** drawn from **posterior distributions** based on the data.

2. **Uncertainty Modeling**:
   - Classical regression provides point predictions without uncertainty.
   - Bayesian regression offers a **distribution over predictions**, capturing both model and parameter uncertainty.

3. **Priors**:
   - Bayesian models incorporate **prior knowledge** by assigning priors (e.g., Normal distributions) to parameters, which is useful in cases of **limited data** or expressing prior beliefs.

---

## 🧩 Model Specification
Given an input feature matrix \( X \) and binary output vector \( Y \), the Bayesian Logistic Regression model is defined as:

$y = \sigma(X \cdot w + b)$

- **$\( w \)$**: Weight vector with a **Normal prior** \( w \sim \mathcal{N}(0, 0.5^2) \)
- **$\( b \)$**: Bias term with a **Normal prior** \( b \sim \mathcal{N}(0, 5^2) \)
- **$\( \sigma \)$**: Sigmoid function mapping linear output to probabilities
- **$\( y \mid x \)$**: Likelihood term with a **Bernoulli** distribution \( y \sim \text{Bernoulli}(\sigma(X \cdot w + b)) \)

This model captures uncertainty by placing priors on weights and bias. Observed data updates the posterior distribution to incorporate new information.

---

## 🔍 Inference using Stochastic Variational Inference (SVI)

In Bayesian models, exact inference is often intractable, especially in high dimensions. **Stochastic Variational Inference (SVI)** approximates the posterior distribution by minimizing **Kullback-Leibler (KL) divergence** between the true posterior and a variational approximation.

### Key SVI Patterns

1. **Variational Parameters**:
   - Instead of learning $\( w \)$ and $\( b \)$ directly, SVI learns **mean and variance of their variational distributions** (e.g., Normal).

2. **Uncertainty Propagation**:
   - The model learns both the **mean** and **uncertainty** of each parameter. Predictions reflect uncertainty by sampling from learned distributions.

3. **Accuracy vs. Uncertainty**:
   - The optimization balances data fit (likelihood) with model complexity (prior regularization), and learned posteriors incorporate both observed data and prior beliefs.

4. **Convergence Patterns**:
   - During training, the **Evidence Lower Bound (ELBO)** is maximized as the objective function. A steady increase in ELBO indicates an optimal posterior approximation.

---

## 🏋️ Training and Loss Behavior

Training Bayesian Logistic Regression involves optimizing the **ELBO** to approximate the posterior distribution of parameters. The loss reflects both data fit and regularization from priors.

### Training Fluctuations

**Loss function** fluctuations are due to:
1. **Posterior Sampling**: Sampling adds randomness to optimization.
2. **KL Divergence Optimization**: Balancing KL divergence complexity can add to fluctuations.
3. **Exploration vs. Exploitation**: Balances exploring uncertain parameter regions and exploiting well-fitting areas.

These fluctuations are expected in **variational inference**. The model typically converges, but training loss may appear noisier than classical regression.

Below is the plot of **loss function over training epochs** showing typical fluctuations:

![Loss Curve](Plots/Loss%20Curve.png)


---

## 📈 Results

Here are the outcomes from **Bayesian Logistic Regression**, with visualizations and comparisons demonstrating the model's uncertainty incorporation.

### Logistic Regression Fit with Data

The plot below shows the data used in the regression task, with a logistic regression line showing the **relationship between feature(s) and target**:

![Logistic Regression Line](Plots/logistic%20regression%20line.png)


---

### Prediction with Uncertainty

This plot includes the **mean logistic regression prediction** with a shaded **95% confidence interval**, capturing uncertainty:

![Prediction with Uncertainty](Plots/Prediction%20with%20Uncertainty.png)


---

### Interpretation

The Bayesian logistic regression model accurately estimates parameters, while **standard deviation** reflects uncertainty. For each parameter:
- **Posterior Mean**: Indicates the most likely value, combining data and prior information.
- **Standard Deviation**: Quantifies parameter uncertainty, providing a confidence range for each estimate.

The confidence intervals around predictions reflect both model uncertainty and potential data variability, making Bayesian models robust for inference in uncertain environments.

---

## 📈 Summary and Conclusion

This project demonstrates **Bayesian Logistic Regression** with Pyro, showing how variational inference estimates parameter uncertainty. Bayesian models allow robust predictions by estimating posterior distributions, combining prior beliefs with data to enhance decision-making in probabilistic settings.

---



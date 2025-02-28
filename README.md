# Probability and Statistics Notes

## 1. Probability Density Function (PDF) & Cumulative Distribution Function (CDF)

## Probability Density Function (PDF)

- Defines the probability distribution of a **continuous** random variable.
- The probability of the variable lying within a range \([a, b]\) is given by:  
  \[
  P(a \leq X \leq b) = \int_{a}^{b} f(x) \,dx
  \]
- The total area under the PDF curve is **1**.
- Example: If \( X \) follows an **exponential distribution** with rate \( \lambda \):
  \[
  f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
  \]


### Cumulative Distribution Function (CDF)

- The probability that the variable takes a value \( \leq x \):
  \[
  F(x) = P(X \leq x)
  \]
- Example: CDF of an **exponential distribution**:
  \[
  F(x) = 1 - e^{-\lambda x}, \quad x \geq 0
  \]

## 2. Continuous Random Variables (CRV)

### Characteristics:

- Can take any real value within an interval.
- Defined using a **PDF**.
- Expected value (mean):
  \[
  E[X] = \int_{-\infty}^{\infty} x f(x) dx
  \]
- Variance:
  \[
  \text{Var}(X) = E[X^2] - (E[X])^2
  \]

### Solving Problems:

- Find **PDF** if given a **CDF**: Differentiate \(F(x)\) to get \(f(x)\).
- Find probabilities using **integration** of PDF.

## 3. Discrete Probability Distributions

### 3.1 Binomial Distribution

- Describes the number of **successes** in **n** independent Bernoulli trials.
- Parameters: **n** (number of trials), **p** (success probability per trial)
- **PMF**:
  \[
  P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}, \quad k = 0, 1, \dots, n
  \]
- Expected value:
  \[
  E[X] = np
  \]
- Variance:
  \[
  \text{Var}(X) = np(1 - p)
  \]

### 3.2 Poisson Distribution

- Models **rare events** occurring in a fixed interval of time or space.
- Parameter: \(\lambda\) (expected number of events per interval)
- **PMF**:
  \[
  P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}, \quad k = 0, 1, 2, \dots
  \]
- Expected value:
  \[
  E[X] = \lambda
  \]
- Variance:
  \[
  \text{Var}(X) = \lambda
  \]

### 3.3 Hypergeometric Distribution

- Models the number of **successes** when selecting a sample **without replacement**.
- Parameters: \( N \) (population size), \( K \) (successes in population), \( n \) (sample size)
- **PMF**:
  \[
  P(X = k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}}
  \]
- Expected value:
  \[
  E[X] = n \frac{K}{N}
  \]

### 3.4 Geometric Distribution

- Models the number of trials until the **first success**.
- **PMF**:
  \[
  P(X = k) = (1 - p)^{k - 1} p, \quad k = 1, 2, 3, \dots
  \]
- Expected value:
  \[
  E[X] = \frac{1}{p}
  \]


## 3. Mean, Variance, Covariance, and Standard Deviation

### Mean (Expected Value)
- The **mean** or **expected value** of a random variable \(X\) is given by:
  \[
  E[X] = \sum x P(X = x) \quad \text{(discrete case)}
  \]
  \[
  E[X] = \int x f(x) dx \quad \text{(continuous case)}
  \]
- Represents the **average value** of the random variable.

### Variance
- Measures the **spread** of the random variable around its mean:
  \[
  \text{Var}(X) = E[X^2] - (E[X])^2
  \]
- **Alternative formula:**
  \[
  \text{Var}(X) = E[(X - E[X])^2]
  \]

### Covariance
- Measures how **two random variables** change together:
  \[
  \text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])]
  \]
- **Properties:**
  - If \( \text{Cov}(X, Y) > 0 \), \(X\) and \(Y\) **increase together**.
  - If \( \text{Cov}(X, Y) < 0 \), \(X\) and \(Y\) **move in opposite directions**.
  - If \( \text{Cov}(X, Y) = 0 \), \(X\) and \(Y\) are **uncorrelated**.

### Standard Deviation
- The **square root** of variance:
  \[
  \sigma_X = \sqrt{\text{Var}(X)}
  \]
- Represents how much values **deviate from the mean** on average.



## 4. Discrete Probability Distributions

### 4.1 Binomial Distribution

- Describes the number of **successes** in **n** independent Bernoulli trials.
- Parameters: **n** (number of trials), **p** (success probability per trial)
- **PMF**:
  \[
  P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}, \quad k = 0, 1, \dots, n
  \]
- Expected value:
  \[
  E[X] = np
  \]
- Variance:
  \[
  \text{Var}(X) = np(1 - p)
  \]

### Independence

- Two events, A and B, are **independent** if the occurrence of one does not affect the probability of the other.
- Mathematically:
  \[
  P(A \cap B) = P(A) P(B)
  \]
- Example: Flipping two coins. The outcome of one coin does not influence the outcome of the other.

### Dependence

- Two events, A and B, are **dependent** if the occurrence of one affects the probability of the other.
- Mathematically:
  \[
  P(A | B) \neq P(A)
  \]
- Example: Drawing cards from a deck **without replacement**. The probability of drawing a certain card changes after each draw.

## 5. Additional Important Theorems in Probability

### Law of Large Numbers

- States that as the number of trials increases, the sample mean converges to the expected value.
- Two types:
  - **Weak Law**: Sample mean converges in probability.
  - **Strong Law**: Sample mean converges almost surely.

### Central Limit Theorem

- States that for a large enough sample size, the sampling distribution of the sample mean approaches a **normal distribution**, regardless of the shape of the population distribution.
- If population mean is \( \mu \) and standard deviation is \( \sigma \), then the sample mean follows:
  \[
  \bar{X} \sim N\left( \mu, \frac{\sigma^2}{n} \right)
  \]

### Gosset's t-Distribution

- Developed by William Sealy Gosset, also known as **Student’s t-distribution**.
- Used when the population variance is unknown and the sample size is small.
- **Formula:**
  \[
  t = \frac{\bar{X} - \mu}{S / \sqrt{n}}
  \]
- Commonly used in hypothesis testing and confidence intervals.

### Slutsky's Theorem

- If a sequence of random variables \( X_n \) converges in probability to a constant \( c \), and another sequence \( Y_n \) has a limiting distribution, then the product \( X_n \cdot Y_n \) has the same limiting distribution as \( c \cdot Y_n \).
- Important in **asymptotic analysis** and **regression theory**.

### Reproductive Theorem

- The **Reproductive Property** states that if independent random variables belong to a particular distribution (e.g., normal, gamma, Poisson), then their sum or a linear transformation of them also belongs to the same distribution.
- **Examples:**
  - If \( X_1, X_2, \dots, X_n \) are **independent normal variables**, then:
    \[
    aX_1 + bX_2 + \dots + cX_n 	ext{ is also normally distributed}
    \]
  - If \( X_1, X_2, \dots, X_n \) follow a **Poisson distribution** with parameters \( \lambda_1, \lambda_2, \dots, \lambda_n \), then:
    \[
    X_1 + X_2 + \dots + X_n \sim 	ext{Poisson}(\lambda_1 + \lambda_2 + \dots + \lambda_n)
    \]
  - The **Gamma distribution** is also closed under summation:
    - If \( X_1, X_2, \dots, X_n \) are **independent gamma-distributed variables** with shape parameters \( k_1, k_2, \dots, k_n \) and a common scale parameter \( 	heta \), then:
      \[
      X_1 + X_2 + \dots + X_n \sim 	ext{Gamma}(k_1 + k_2 + \dots + k_n, 	heta)
      \]

### Gamma Distribution

- The **Gamma distribution** is a two-parameter family of continuous probability distributions.
- **Parameters:**
  - \( k \) (shape parameter)
  - \( 	heta \) (scale parameter)
- **Probability Density Function (PDF):**
  \[
  f(x) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}, \quad x > 0
  \]
- **Expected Value:**
  \[
  E[X] = k\theta
  \]
- **Variance:**
  \[
  \text{Var}(X) = k\theta^2
  \]
- **Special Cases:**
  - When \( k = 1 \), the Gamma distribution reduces to the **Exponential distribution**.
  - When \( k \) is an integer, the Gamma distribution represents the sum of \( k \) **independent exponential** random variables with mean \( 	heta \).



# Probability and Statistics Notes 

## 1. Probability Density Function (PDF) & Cumulative Distribution Function (CDF) 


### Probability Density Function (PDF) - Defines the probability distribution of a **continuous** random variable. 


- The probability of the variable lying within a range \([a, b]\) is given by: $$ P(a \leq X \leq b) = \int_{a}^{b} f(x)\,dx $$ 

- The total area under the PDF curve is **1**. 

- **Example:** If \(X\) follows an exponential distribution with rate \(\lambda\): $$ f(x) = \lambda e^{-\lambda x}, \quad x \ge 0 $$



 ### Cumulative Distribution Function (CDF)

- Gives the probability that the variable takes a value \(\leq x\): $$ F(x) = P(X \le x) $$


- For continuous random variables: $$ F(x) = \int_{-\infty}^{x} f(t)\,dt $$ - **Example:** CDF of an exponential distribution: $$ F(x) = 1 - e^{-\lambda x}, \quad x \ge 0 $$ 

## 2. Continuous Random Variables (CRV) 

### Characteristics 
- Can take any real value within an interval. - Defined using a **PDF**. - 


**Expected value (mean):** $$ E[X] = \int_{-\infty}^{\infty} x\,f(x)\,dx $$ 

 **Variance:** $$ \operatorname{Var}(X) = E[(X - E[X])^2] = \int_{-\infty}^{\infty} (x - E[X])^2\,f(x)\,dx $$ 


### Solving Problems 

- **Finding PDF from CDF:** Differentiate \( F(x) \) to get \( f(x) \). 
-  **Calculating probabilities:** Use integration of the PDF.

### 3.1 Binomial Distribution 

- Describes the number of **successes** in \( n \) independent Bernoulli trials.
 - **Parameters:** \( n \) (number of trials), \( p \) (success probability per trial). 
 - **Probability Mass Function (PMF):** $$ P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} $$ 
 - **Expected value:** \( E[X] = np \)
 
 - **Variance:** \( \operatorname{Var}(X) = np(1-p) \)
 
 ### Probability Mass Function (PMF) 

$$ P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} $$ 
 
 - **Expected value:** \( E[X] = \lambda \)


#### Mean, Std, Variance, and Covariance 
**1. Mean (\(\mu\))** The mean (or average) of a set of \(N\) data points \(x_1, x_2, \dots, x_N\) is: 

$$ \mu = \frac{\sum_{i=1}^{N} x_i}{N} $$ 

**2. Variance (\(\sigma^2\))** The variance measures how spread out the data is from the mean: $$ \sigma^2 = \frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N} $$ 

**3. Standard Deviation (\(\sigma\))** The standard deviation is the square root of the variance: $$ \sigma = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}} $$ 

**4. Covariance** Covariance measures how two variables \(X\) and \(Y\) change together: $$ \mathrm{Cov}(X, Y) = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{N} $$ 

### 3.2 Poisson Distribution 
- Models **rare events** occurring in a fixed interval of time or space. 

**Parameter:** \( \lambda \) (expected number of events per interval). 

**PMF:** $$ P(X=k) = \frac{e^{-\lambda} \lambda^k}{k!} $$ - 
**Expected value:** \( E[X] = \lambda \) 

- **Variance:** \( \operatorname{Var}(X) = \lambda \)

 ### 3.3 Hypergeometric Distribution 
 - Models the number of **successes** when selecting a sample **without replacement**.
 - **Parameters:** \( N \) (population size), \( K \) (number of successes in the population), \( n \) (sample size). 

 - **PMF:** $$ P(X=k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}} $$
- **Expected value:** $$ E[X] = n\frac{K}{N} $$ 
### 3.4 Geometric Distribution 
- Models the number of trials until the **first success**. 

- **PMF:** $$ P(X=k) = (1-p)^{k-1}p $$

- **Expected value:** $$ E[X] = \frac{1}{p} $$ 

## 4. Additional Important Theorems in Probability 

### Law of Large Numbers 

- As the number of trials increases, the sample mean converges to the expected value. 

- **Types:** 
- **Weak Law:** Convergence in probability. 
- **Strong Law:** Almost sure convergence.

### Central Limit Theorem
- For a large enough sample size, the sampling distribution of the sample mean approaches a normal distribution regardless of the population distribution. 

- If the population mean is \( \mu \) and standard deviation is \( \sigma \), then: $$ \bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) $$ - 

- When population variance is unknown, 
- **Student's t-distribution** is used. 

### Reproductive Theorem
- If independent random variables belong to a particular distribution (e.g., normal, gamma, Poisson), then their sum or any linear combination also follows the same type of distribution. 

- **Example:** If \( X_1, X_2, \dots, X_n \) are independent normal variables, then any linear combination is normally distributed. 

### Contributions of Gosset and Slutsky 

#### Gosset (Student's t-Distribution) 
- Introduced the **t-distribution** for small sample sizes when the population variance is unknown. 
- **t-statistic:** $$ t = \frac{\bar{X} - \mu}{s/\sqrt{n}} $$ 
- Widely used in hypothesis testing and confidence intervals

#### Slutsky's Theorem 
- If a sequence of random variables converges in probability to a constant \( c \), and another sequence has a limiting distribution, then the product converges in distribution to \( c \) times the limiting distribution of the other sequence.

- Important in asymptotic analysis and regression theory. 
## 5. Summary - **PDF/CDF:** 

Fundamental for continuous variables. 
- **Discrete Distributions:** Includes Binomial, Poisson, Geometric, and Hypergeometric.
- **Law of Large Numbers:** Ensures convergence of sample mean. 
- **Central Limit Theorem:** Explains normality of sample means.  
- **Reproductive Theorem:** Maintains distribution consistency under linear transformations. 
- **Gosset's t-distribution:** Crucial for small sample inference. 
- **Slutsky's theorem:** Aids in asymptotic analysis. 
- **Counting Principles and Sampling Distributions:** Key in inferential statistics. 

## 1. Probability Fundamentals ### Basic Probability Concepts 

- **Sample Space (\(\Omega\))**: Set of all possible outcomes. 
- **Event:** A subset of the sample space. 

### Probability Axioms 
- \(0 \le P(A) \le 1\) for any event \(A\). 
- \(P(\Omega) = 1\). 
- For mutually exclusive events: 
$$ P(A \cup B) = P(A) + P(B) $$ 

### Combinations and Permutations 
- **Permutations (order matters):** $$ P(n, r) = \frac{n!}{(n-r)!} $$
- **Combinations (order doesn't matter):** $$ C(n, r) = \frac{n!}{r!(n-r)!} $$ 
### Conditional Probability & Independence 

- **Conditional Probability:** $$ P(A|B) = \frac{P(A \cap B)}{P(B)} $$ 
- **Independence:** \(A\) and \(B\) are independent if: $$ P(A \cap B) = P(A) \cdot P(B) $$ 
- **Law of Total Probability:** $$ P(A) = \sum_i P(A|B_i)P(B_i) $$ 

### Bayes' Theorem - **Formula:** $$ P(A|B) = \frac{P(B|A) \times P(A)}{P(B)} $$ 

- **Components:** 
	- \(P(A)\): Prior probability. 
	- \(P(B|A)\): Likelihood. 
	- \(P(A|B)\): Posterior probability. 
## 2. Random Variables and Distributions ### Random Variables 
 **Definition:** A function that maps outcomes to real numbers. 
 - **Types:** 
	 - **Discrete:** Takes countable values.
	 - **Continuous:** Takes uncountable values. 
	 ### Expectation and Moments
	 #### Expected Value (Mean) 
	 - **Discrete:** $$ E[X] = \sum x\,P(X=x) $$ 
	 - **Continuous:** $$ E[X] = \int_{-\infty}^{\infty} x\,f(x)\,dx $$ 
	 - **Properties:** $$ E[aX + b] = a\,E[X] + b $$ $$ E[X + Y] = E[X] + E[Y] $$ 
#### Variance and Standard Deviation 
- **Variance:** $$ \operatorname{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2 $$
- **Standard Deviation:** $$ \sigma = \sqrt{\operatorname{Var}(X)} $$ 
- **Properties:** $$ \operatorname{Var}(aX + b) = a^2\,\operatorname{Var}(X) $$ 

- For independent \(X\) and \(Y\): $$ \operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) $$ 

#### Covariance and Correlation - **Covariance:** $$ \operatorname{Cov}(X, Y) = E[(X-\mu_X)(Y-\mu_Y)] = E[XY] - E[X]E[Y] $$

- **Correlation Coefficient:** $$ \rho = \frac{\operatorname{Cov}(X,Y)}{\sigma_X \sigma_Y} $$
- \(-1 \le \rho \le 1\) - \(\rho = \pm1\) indicates a perfect linear relationship. - \(\rho = 0\) indicates no linear relationship. 

#### Moment Generating Functions (MGF) 

- **Definition:** $$ M_X(t) = E[e^{tX}] $$ 
- **Properties:** 
	- Uniquely determines the distribution. 
	- Moments can be derived as: $$ E[X^n] = \frac{d^n}{dt^n}M_X(t)\Big|_{t=0} $$ 

- For independent variables: $$ M_{X+Y}(t) = M_X(t) \cdot M_Y(t) $$ 

## 3. Discrete Probability Distributions 
### Bernoulli Distribution 
- Models a single trial with success probability \(p\). - **PMF:** $$ P(X=1)=p,\quad P(X=0)=1-p $$ 
- **Expected Value:** \( E[X]=p \) 
- **Variance:** \( \operatorname{Var}(X)=p(1-p) \) 

 - **PDF:** $$ f(x)=\frac{1}{b-a},\quad a \le x \le b $$ - **Expected Value:** $$ E[X]=\frac{a+b}{2} $$ - **Variance:** $$ \operatorname{Var}(X)=\frac{(b-a)^2}{12} $$ 
 - ### Normal (Gaussian) Distribution - **PDF:** $$ f(x)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) $$ 
 - **Standard Normal:** \( Z \sim N(0,1) \) 
 - **Transformation:** \( Z = \frac{X-\mu}{\sigma} \) 
 - **68-95-99.7 Rule:** Approximately \(68\%\), \(95\%\), and \(99.7\%\) of data lie within 1, 2, and 3 standard deviations, respectively. 
 - ### Exponential Distribution - **PDF:** $$ f(x)=\lambda e^{-\lambda x}, \quad x \ge 0 $$ - **Expected Value:** $$ E[X]=\frac{1}{\lambda} $$ - **Variance:** $$ \operatorname{Var}(X)=\frac{1}{\lambda^2} $$ - **Memoryless Property:** $$ P(X>s+t \mid X>s)=P(X>t) $$
 
 ### Gamma Distribution 
 - **Parameters:** \( \alpha \) (shape) and \( \beta \) (scale). 
 - **PDF:** $$ f(x)=\frac{x^{\alpha-1}e^{-x/\beta}}{\beta^\alpha\Gamma(\alpha)} $$ - **Expected Value:** $$ E[X]=\alpha\beta $$ - **Variance:** $$ \operatorname{Var}(X)=\alpha\beta^2 $$ 

### Beta Distribution - **Models:** Probabilities or proportions. - **PDF:** $$ f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)},\quad 0\le x\le 1 $$ - **Expected Value:** $$ E[X]=\frac{\alpha}{\alpha+\beta} $$ - **Variance:** $$ \operatorname{Var}(X)=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)} $$ 

### t-Distribution 
- Used when estimating the mean with unknown population variance.
- **Parameter:** \( \nu \) (degrees of freedom). 
- Approaches the normal distribution as \( \nu \) increases. 
- Widely used in hypothesis testing and confidence intervals. 

### Chi-Square Distribution
- Sum of squared standard normal variables. 
- **Parameter:** \( k \) (degrees of freedom). 
- **Expected Value:** \( E[X]=k \)
- **Variance:** \( \operatorname{Var}(X)=2k \) 
### F-Distribution 
- Ratio of two chi-square distributed variables. 

- **Parameters:** \( d_1 \) and \( d_2 \) (degrees of freedom).
- Commonly used in ANOVA and variance testing. ---
 
## 5. Joint Distributions 

### Joint Probability Distributions 
- Describes the distribution of two or more random variables. 
- **Joint CDF:** $$ F(x,y)=P(X\le x,\; Y\le y) $$ 
- **Joint PMF (discrete):** $$ p(x,y)=P(X=x,\; Y=y) $$
- **Joint PDF (continuous):** $$ \iint f(x,y)\,dx\,dy=1 $$ 

### Marginal Distributions

- Derived by summing or integrating out other variables. 

- **Discrete:** $$ P_X(x)=\sum_y P(X=x, Y=y) $$

- **Continuous:** $$ f_X(x)=\int f(x,y)\,dy $$

### Conditional Distributions 

- **Discrete:** $$ P(X=x|Y=y)=\frac{P(X=x,Y=y)}{P(Y=y)} $$ 

- **Continuous:** $$ f(x|y)=\frac{f(x,y)}{f_Y(y)} $$ 

### Independence of Random Variables

- \(X\) and \(Y\) are independent if: $$ f(x,y)=f_X(x)\cdot f_Y(y) \quad \text{for all } x,y $$

- For independent variables: 
-  \(E[XY]=E[X]E[Y]\) - \(\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)\) --- 

## 6. Sampling Distributions and Limit Theorems 

### Sampling Distribution
- The distribution of a statistic (e.g., sample mean \(\bar{X}\), sample variance \(S^2\), or sample proportion \(\hat{p}\)) computed from random samples. 

### Law of Large Numbers 
- As the number of trials increases, the sample mean converges to the expected value. 
- **Types:** 
- **Weak Law:** Convergence in probability. 
- **Strong Law:** Almost sure convergence. 

### Central Limit Theorem
- For a large sample size, the sampling distribution of the sample mean is approximately normal.

- If the population has mean \( \mu \) and standard deviation \(\sigma \), then: $$ \bar{X}\sim N\Bigl(\mu,\frac{\sigma^2}{n}\Bigr) $$ - 

When the population variance is unknown, the Student's t-distribution is used. 

### Reproductive Theorem

- If independent random variables belong to a particular distribution, then their sum or any linear combination also follows that distribution.

- **Example:** If \( X_1, X_2, \dots, X_n \) are independent normal variables, then $$ aX_1 + bX_2 + \cdots + cX_n $$ is normally distributed. 

### Slutsky's Theorem 
- If a sequence \(X_n\) converges in probability to a constant \(c\), and another sequence \(Y_n\) has a limiting distribution, then the product \(X_n \cdot Y_n\) converges in distribution to \(c \cdot Y_n\). 

- This theorem is important in asymptotic analysis and regression theory. --- 

## 7. Parameter Estimation 

### Point Estimation 
- A method for estimating a population parameter with a single value.

- **Properties of good estimators:** 
- **Unbiasedness:** \(E[\hat{\theta}]=\theta\)
- **Consistency:** \(\hat{\theta} \to \theta\) as \(n\to\infty\) 
- **Efficiency:** Minimum variance among unbiased estimators. 
- **Sufficiency:** Contains all information about the parameter. 

### Maximum Likelihood Estimation (MLE)
- **Method:** Find the parameter \( \theta \) that maximizes the likelihood \( L(\theta|x) \). 
- **Log-likelihood:** $$ \ell(\theta|x)=\log\bigl(L(\theta|x)\bigr) $$

- **Properties:** Consistency, asymptotic normality, and efficiency. - **Note:** Closely related to the Kullback-Leibler divergence in information theory. 

### Method of Moments 
- Estimate parameters by equating sample moments to population moments. 
- Simpler but often less efficient than MLE. 

### Sufficient Statistics 
- A statistic is sufficient if it captures all information in the sample about the parameter. 
### **Fisher-Neyman Factorization Theorem:** 
- Provides a method to determine sufficient statistics. 
- Closely related to exponential family distributions. 

## 8. Interval Estimation and Hypothesis Testing 

### Confidence Intervals 
- An interval that, with a specified confidence level \((1-\alpha)\), is likely to contain the true parameter. 
- Construction methods vary based on the parameter and distribution. 

### Hypothesis Testing 
- **Setup:** Compare a null hypothesis (\(H_0\)) against an alternative hypothesis (\(H_1\)).
- **Test Statistic:** A measure to evaluate the evidence against \(H_0\). - **Errors:**
- **Type I Error:** Rejecting \(H_0\) when it is true (\(\alpha\)).
- **Type II Error:** Failing to reject \(H_0\) when it is false (\(\beta\)). 
- **Power of the Test:** \(1-\beta\) (the probability of correctly rejecting a false \(H_0\)). 
-  **p-value:** The probability of observing a test statistic as extreme or more extreme than the one observed, assuming \(H_0\) is true. - **Common Tests:** 
	- **z-test:** When the population variance is known. 
	- **t-test:** When the population variance is unknown (one-sample, two-sample, or paired). 
	- **F-test:** For comparing variances.
	- **Chi-square test:** For goodness-of-fit and testing independence. 
- **ANOVA:** Analysis of variance for comparing multiple means. --- 

## 9. Stochastic Processes 

### Markov Chains
- A stochastic process with the Markov (memoryless) property. 

- **Transition Probabilities:** $$ P(X_{n+1}=j \mid X_n=i)=p_{ij} $$ 

- **Transition Matrix:** \(P=[p_{ij}]\) 
- **States:** Classified as transient, recurrent, or absorbing. 
- **Stationary Distribution:** Long-run behavior of the chain.
- **Applications:** Queueing theory, genetics, economics. 

### Poisson Process 
- A counting process for random events over time or space. 

**Properties:** 
- Independent and stationary increments. 
- \( N(t) \sim \text{Poisson}(\lambda t) \) 
- Interarrival times are exponentially distributed with rate \(\lambda\). 

### Brownian Motion 
- A continuous-time stochastic process with continuous paths. 
- **Properties:** 

- \(B(0)=0\). - Independent increments. - Normal increments: $$ B(t)-B(s) \sim N(0,t-s) $$ 

### Order Statistics 

- Arranging sample values in ascending order: $$ X_{(1)} \le X_{(2)} \le \cdots \le X_{(n)} $$ 

- **Applications:** Reliability theory, extreme value analysis. 

### Bayesian Statistics 

- Uses Bayes' theorem to update probabilities based on new evidence. 

**Process:** 
- Prior \(\to\) Likelihood \(\to\) Posterior. 

**Advantages:** 
- Incorporates prior knowledge and provides direct probability statements about parameters. 

**Bayesian Estimation:** 
- Yields credible intervals. 


### Regression Analysis 

- **Linear Regression:** 
 \( Y = \beta_0 + \beta_1X + \epsilon \)

- **Multiple Regression:** 
$$ Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_kX_k + \epsilon $$ 

- **Estimation:** 
Typically via least squares. 
- **Assumptions:** Linearity, independence of errors, homoscedasticity, and normality of errors. 

### Time Series Analysis 

- Data collected sequentially over time. 
**Components:** Trend, seasonality, cyclical, and irregular. 
**Models:**  AR, MA, ARIMA, GARCH. 
**Applications:** Forecasting and analyzing temporal patterns. ---

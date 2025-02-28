Probability and Statistics Notes
1. Probability Fundamentals
Basic Probability Concepts

Sample Space (Ω): Set of all possible outcomes
Event: Subset of the sample space
Probability Axioms:

0 ≤ P(A) ≤ 1 for any event A
P(Ω) = 1
For mutually exclusive events: P(A ∪ B) = P(A) + P(B)


Combinations and Permutations:

Permutations (order matters): P(n,r) = n!/(n-r)!
Combinations (order doesn't matter): C(n,r) = n!/(r!(n-r)!)



Conditional Probability & Independence

Conditional Probability: P(A|B) = P(A ∩ B)/P(B)
Independence: Events A and B are independent if P(A ∩ B) = P(A) · P(B)
Law of Total Probability: P(A) = P(A|B₁)P(B₁) + P(A|B₂)P(B₂) + ... + P(A|Bₙ)P(Bₙ)

Bayes' Theorem

Formula: P(A|B) = [P(B|A) × P(A)]/P(B)
Components:

P(A): Prior probability
P(A|B): Posterior probability
P(B|A): Likelihood


Applications: Medical diagnostics, spam filtering, machine learning

2. Random Variables and Distributions
Random Variables

Definition: Function that maps outcomes to real numbers
Types:

Discrete: Takes countable values
Continuous: Takes uncountable values



Probability Density Function (PDF) & Cumulative Distribution Function (CDF)
Probability Density Function (PDF)

Defines the probability distribution of a continuous random variable
The probability of the variable lying within a range [a, b] is given by: P(a ≤ X ≤ b) = ∫ᵃᵇ f(x) dx
The total area under the PDF curve is 1
Example: If X follows an exponential distribution with rate λ: f(x) = λe^(-λx) for x ≥ 0

Cumulative Distribution Function (CDF)

The probability that the variable takes a value ≤ x: F(x) = P(X ≤ x)
For continuous RVs: F(x) = ∫₍₋∞₎ˣ f(t) dt
Example: CDF of an exponential distribution: F(x) = 1 - e^(-λx) for x ≥ 0

Expectation and Moments
Expected Value (Mean)

Discrete: E[X] = Σ x·P(X=x)
Continuous: E[X] = ∫ x·f(x) dx
Properties:

E[aX + b] = a·E[X] + b
E[X + Y] = E[X] + E[Y]



Variance and Standard Deviation

Variance: Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
Standard Deviation: σ = √Var(X)
Properties:

Var(aX + b) = a²·Var(X)
For independent X and Y: Var(X + Y) = Var(X) + Var(Y)



Covariance and Correlation

Covariance: Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] = E[XY] - E[X]E[Y]
Correlation coefficient: ρ = Cov(X,Y)/(σₓσᵧ)
Properties:

-1 ≤ ρ ≤ 1
ρ = ±1 indicates perfect linear relationship
ρ = 0 indicates no linear relationship (but potentially nonlinear)



Moment Generating Functions (MGF)

Definition: M₍X₎(t) = E[e^(tX)]
Properties:

Uniquely determines the distribution
Can derive moments: E[X^n] = d^n/dt^n M₍X₎(t)|ₜ₌₀
MGF of sum of independent variables: M₍X+Y₎(t) = M₍X₎(t) · M₍Y₎(t)



3. Discrete Probability Distributions
Bernoulli Distribution

Models a single trial with success probability p
PMF: P(X=1) = p, P(X=0) = 1-p
E[X] = p
Var(X) = p(1-p)

Binomial Distribution

Describes the number of successes in n independent Bernoulli trials
Parameters: n (number of trials), p (success probability per trial)
PMF: P(X=k) = C(n,k) · p^k · (1-p)^(n-k)
E[X] = np
Var(X) = np(1-p)

Poisson Distribution

Models rare events occurring in a fixed interval of time or space
Parameter: λ (expected number of events per interval)
PMF: P(X=k) = (e^(-λ) · λ^k)/k!
E[X] = λ
Var(X) = λ
Approximates binomial when n is large and p is small (λ = np)

Geometric Distribution

Models the number of trials until the first success
Parameter: p (success probability)
PMF: P(X=k) = (1-p)^(k-1) · p
E[X] = 1/p
Var(X) = (1-p)/p²

Negative Binomial Distribution

Number of trials until r successes
PMF: P(X=k) = C(k-1,r-1) · p^r · (1-p)^(k-r)
E[X] = r/p
Var(X) = r(1-p)/p²

Hypergeometric Distribution

Models the number of successes when selecting a sample without replacement
Parameters: N (population size), K (successes in population), n (sample size)
PMF: P(X=k) = [C(K,k) · C(N-K,n-k)]/C(N,n)
E[X] = n·K/N
Var(X) = n·K/N·(N-K)/N·(N-n)/(N-1)

4. Continuous Probability Distributions
Uniform Distribution

Constant probability over interval [a,b]
PDF: f(x) = 1/(b-a) for a ≤ x ≤ b
E[X] = (a+b)/2
Var(X) = (b-a)²/12

Normal (Gaussian) Distribution

Bell-shaped distribution defined by mean μ and variance σ²
PDF: f(x) = (1/√(2πσ²)) · e^(-(x-μ)²/(2σ²))
Standard normal: Z ~ N(0,1)
Transformation: Z = (X-μ)/σ
68-95-99.7 rule: Probabilities within 1,2,3 standard deviations

Exponential Distribution

Models time between events in a Poisson process
Parameter: λ (rate parameter)
PDF: f(x) = λe^(-λx) for x ≥ 0
E[X] = 1/λ
Var(X) = 1/λ²
Memoryless property: P(X > s+t | X > s) = P(X > t)

Gamma Distribution

Generalization of exponential and chi-squared distributions
Parameters: α (shape) and β (scale)
PDF: f(x) = (x^(α-1) · e^(-x/β))/(β^α · Γ(α))
E[X] = αβ
Var(X) = αβ²

Beta Distribution

Models probabilities or proportions
Parameters: α and β (shape parameters)
PDF: f(x) = [x^(α-1) · (1-x)^(β-1)]/B(α,β) for 0 ≤ x ≤ 1
E[X] = α/(α+β)
Var(X) = αβ/[(α+β)²(α+β+1)]

t-Distribution

Used when estimating mean with unknown population variance
Parameter: v (degrees of freedom)
Approaches normal distribution as v increases
Used in hypothesis testing and constructing confidence intervals

Chi-Square Distribution

Sum of squared standard normal random variables
Parameter: k (degrees of freedom)
PDF: f(x) = (x^(k/2-1) · e^(-x/2))/(2^(k/2) · Γ(k/2))
E[X] = k
Var(X) = 2k

F-Distribution

Ratio of two chi-square distributions
Parameters: d₁, d₂ (degrees of freedom)
Used in ANOVA and testing equality of variances

5. Joint Distributions
Joint Probability Distributions

Describes the probability distribution of two or more random variables
Joint CDF: F(x,y) = P(X ≤ x, Y ≤ y)
Joint PMF (discrete): p(x,y) = P(X=x, Y=y)
Joint PDF (continuous): ∫∫ f(x,y) dx dy = 1

Marginal Distributions

Obtained by summing/integrating out other variables
For discrete: P₍X₎(x) = Σᵧ P(X=x, Y=y)
For continuous: f₍X₎(x) = ∫ f(x,y) dy

Conditional Distributions

Discrete: P(X=x|Y=y) = P(X=x, Y=y)/P(Y=y)
Continuous: f(x|y) = f(x,y)/f₍Y₎(y)

Independence of Random Variables

X and Y are independent if f(x,y) = f₍X₎(x) · f₍Y₎(y) for all x,y
For independent variables: E[XY] = E[X]E[Y]
For independent variables: Var(X+Y) = Var(X) + Var(Y)

6. Sampling Distributions and Limit Theorems
Sampling Distribution

Distribution of a statistic computed from random samples
Important examples:

Sample mean (X̄)
Sample variance (S²)
Sample proportion (p̂)



Law of Large Numbers

States that as the number of trials increases, the sample mean converges to the expected value
Two types:

Weak Law: Sample mean converges in probability
Strong Law: Sample mean converges almost surely



Central Limit Theorem

States that for a large enough sample size, the sampling distribution of the sample mean approaches a normal distribution, regardless of the shape of the population distribution
If population mean is μ and standard deviation is σ, then the sample mean follows:
X̄ ~ N(μ, σ²/n) for large n
When population variance is unknown, Student's t-distribution is used instead

Reproductive Theorem

If independent random variables belong to a particular distribution (e.g., normal, gamma, Poisson), then their sum or a linear transformation of them also belongs to the same distribution
Example: If X₁, X₂, ..., Xₙ are independent normal variables, then:
aX₁ + bX₂ + ... + cXₙ is also normally distributed

Slutsky's Theorem

If a sequence of random variables Xₙ converges in probability to a constant c, and another sequence Yₙ has a limiting distribution, then the product Xₙ·Yₙ has the same limiting distribution as c·Yₙ
Important in asymptotic analysis and regression theory

7. Parameter Estimation
Point Estimation

Method for estimating a population parameter using a single value
Properties of good estimators:

Unbiasedness: E[θ̂] = θ
Consistency: θ̂ → θ as n → ∞
Efficiency: Minimum variance
Sufficiency: Contains all information about parameter



Maximum Likelihood Estimation (MLE)

Method for parameter estimation
MLE: θ̂ = argmax L(θ|x)
Log-likelihood: ℓ(θ|x) = log(L(θ|x))
Properties:

Consistency
Asymptotic normality
Efficiency


Connection to information theory via Kullback-Leibler divergence

Method of Moments

Equate sample moments with population moments
Simpler but often less efficient than MLE

Sufficient Statistics

Contains all information in sample about parameter
Fisher-Neyman Factorization Theorem
Minimal sufficient statistics
Relationship with exponential families

8. Interval Estimation and Hypothesis Testing
Confidence Intervals

Interval estimation for parameters
Interpretation: contains true parameter with specified confidence level (1-α)
Construction methods for different parameters
Relationship with hypothesis testing

Hypothesis Testing

Null hypothesis (H₀) vs. Alternative hypothesis (H₁)
Test statistic: Measure to evaluate evidence against H₀
Type I error: Rejecting H₀ when it's true (probability = α)
Type II error: Failing to reject H₀ when it's false (probability = β)
Power of a test: 1-β (probability of correctly rejecting false H₀)
p-value: Probability of observing test statistic as extreme or more extreme than observed, assuming H₀ is true

Common Statistical Tests

z-test: Used when population variance is known
t-test: Used when population variance is unknown

One-sample t-test
Two-sample t-test (independent samples)
Paired t-test (dependent samples)


F-test: Tests ratio of variances
Chi-square test:

Goodness-of-fit test
Test of independence


ANOVA: Analysis of variance (comparing multiple means)

9. Stochastic Processes
Markov Chains

Stochastic process with Markov property (memoryless)
Transition probabilities: P(X₍n+1₎ = j | X₍n₎ = i) = p₍ij₎
Transition matrix P = [p₍ij₎]
Classification of states: transient, recurrent, absorbing
Stationary distribution
Applications in queueing theory, genetics, and economics

Poisson Process

Count process for events occurring randomly over time/space
Properties:

Independent increments
Stationary increments
N(t) ~ Poisson(λt)
Interarrival times are exponential with rate λ



Brownian Motion

Continuous-time stochastic process
Properties:

Continuous paths
Independent increments
Normal increments: B(t) - B(s) ~ N(0, t-s)
B(0) = 0


Applications in finance, physics, and biology

10. Advanced Topics
Order Statistics

Arranging sample values in ascending order: X₍₁₎ ≤ X₍₂₎ ≤ ... ≤ X₍n₎
Distribution of minimum, maximum, median, and other order statistics
Applications in reliability theory and extreme value analysis

Bayesian Statistics

Uses Bayes' theorem to update probability as more evidence becomes available
Prior distribution → Likelihood → Posterior distribution
Bayesian estimation and credible intervals
Advantages:

Incorporates prior knowledge
Provides direct probability statements about parameters



Regression Analysis

Linear regression: Y = β₀ + β₁X + ε
Multiple regression: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε
Least squares estimation
Assumptions:

Linearity
Independence of errors
Homoscedasticity
Normality of errors



Time Series Analysis

Data collected over time
Components: Trend, Seasonality, Cyclical, Irregular
Models:

Autoregressive (AR)
Moving Average (MA)
ARIMA
GARCH


Forecasting methods

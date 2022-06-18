# Anamoly Detection

## Algorithm
Training set: $\{ x^{(1)}, \dots, x^{(m)} \}$
Each example $x^{(i)} \in \mathbb{R}^n$

Now, we assume all the variables follow the Guassian Distribution
$x_j \sim \mathcal{N}(\mu_j, \sigma_j^2) \quad \forall \quad j$
Now, Probability of x occuring
$$
\begin{align*}
p(x) & = p(x_1, \mu_1, \sigma_1^2) \cdot p(x_2, \mu_2, \sigma_2^2) \dots p(x_n, \mu_n, \sigma_n^2) \\
& = \prod_{j=1}^n p(x_j, \mu_j, \sigma_j^2) \\
& = \prod_{j=1}^n \frac{1}{\sqrt{2\pi} \sigma_j} \exp
\left( - \frac{(x_j-\mu_j)^2}{2\sigma_j^2} \right)
\end{align*}
$$

1. Choosing features $x_j$ which are thought to be the indicative of the anamolies
2. Fit parameters $\mu_i$ amd $\sigma_j$ for all $j$
	$$
	\begin{align*}
		\mu_j & = \frac{1}{m} \sum_{i=1}^m x_j^{(i)} \\
		\sigma_j^2 & =\frac{1}{m-1} \sum_{i=1}^m  (x_j-\mu_j)^2 \\
	\end{align*}
	$$
3. Given the new example $x$, compute $p(x)$ using
	$$
	p(x) = \prod_{j=1}^n \frac{1}{\sqrt{2\pi} \sigma_j} \exp
	\left( - \frac{(x_j^{(i)}-\mu_j)^2}{2\sigma_j^2} \right)
	$$
	Classify $x$ as anomaly if $p(x) < \epsilon$

(Note: Dividing by $m$ is common practise in ML to calculate $\sigma_j^2$)
(Note: Independence is assumed while calculating $p(x)$ and this usually works in ML)

## Developing and Evaluating an anamoly detection system

**The importance of real-number evaluation**: When developing a learning algorithm(choosing features, etc... ),  making decisions is much easier if we have a way of evaluating our learning algorithm. 

### Aircraft engines motivating example
10000 : Good (normal) engines
20 : Flawed Engines

Training Set: 6000 good engines
CV: 2000 good engines ($y=0$) and 10 anamolous ($y=1$)
Test: 2000 good engines ($y=0$) and 10 anamolous ($y=1$)

### Algorithm Evaluation
- Fit model $p(x)$ on training set $\{ x^{(1)}, \dots, x^{(m)}\}$
- On the cross validation / test set, predict
	$$
	y = \begin{cases}
		1 & \text{if } p(x) < \epsilon \ (\text{anamaly}) \\
		0 & \text{if } p(x) \geq \epsilon \ (\text{normal}) \\
	\end{cases}
	$$
- Possible evaluation metrics
	* True positive, false positive, false negative, true negative
	* Precision, recall
	* F~1~ score
- Cross Validation set can also be used to choose parameter $\epsilon$

## Anamoly Detection vs Superwised Learning

### Anamoly Detection is suitable in the following cases
- Very small number of positive examples. $(y=1)$ (0-20 is  common). 
- Large number of negative examples. $(y=0)$
- Many different “types” of anomalies. Hard for any algorithm to learn from positive examples what the anomalies look like.
-  Future anomalies may look nothing like any of the anomalous  examples we’ve seen so far. 
- Fraud detection
- Manufacturing (e.g., aircraft engines)
- Monitoring machines in a data center

### Superwised Learning is suitable in the following cases
- Large number of positive and negative examples
- Enough positive examples for  algorithm to get a sense of what  positive examples are like,  future  positive examples likely to be  similar to ones in training set
- Email spam classification
- Weather prediction
- Cancer classification

<br> <br>

> ### Choosing what features to use 
> - Converting the distribution to Gaussion can be tried by taking square roots or logs or exponents. 
> - Choosing features that take unusually large or unusually small value. (Example: While monitoring computers in a data center, (CPU Load / Network Traffic) blows up when CPU load is high and network traffic is low


## Multivariable Guassian Distribution
$x \in \mathbb{R}^n$. Don't model $p(x_i)$ seperately.
Model $p(x)$ in one go

$$
\text{Parameters: } \quad \mu \in \mathbb{R}^n, \quad
\Sigma \in \mathbb{R}^{n \times n} \\
p(x; \mu, \Sigma) = 
\frac{1}{(2 \pi)^\frac{n}{2} |\Sigma|^\frac{n}{2}}
\exp \left( - \frac{1}{2} (x - \mu)^T \ \Sigma^{-1} \ (x - \mu)
\right)
$$

### Anomaly detection using multivariable distribution
1. Fit parameters $\mu_i$ amd $\sigma_j$ for all $j$
	$$
	\begin{align*}
		\mu & = \frac{1}{m} \sum_{i=1}^m x^{(i)} \\
		\Sigma & =\frac{1}{m-1} \sum_{i=1}^m
			(x^{(i)} - \mu) (x^{(i)} - \mu)^T \\
	\end{align*}
	$$
2. Given the new example $x$, compute $p(x)$ using
	$$
	p(x) = \frac{1}{(2 \pi)^\frac{n}{2} |\Sigma|^\frac{n}{2}}
\exp \left( - \frac{1}{2} (x - \mu)^T \ \Sigma^{-1} \ (x - \mu)
\right)
	$$
3. Flag $x$ as anomaly if $p(x) < \epsilon$

### Differences between the old and the new model
|Original Model| Multivariable Model |
|--|--|
|Manually create features to capture corelations | Automatically captures corelations between features |
| Computationally cheaper | Computationally more expensive |
| OK even if m (training set size) is small | Must have m > n or else $\Sigma$ will be non-invertible. Typically have $m \geq 10n$

# Recommender Systems

- $n_u :$ Number of users
- $n_m :$ Number of movies
- $r(i, j) = 1$ if user $j$ has rated movie $i$ $(0$ otherwise $)$
- $y^{i, j} :$ The rating given by user $j$ to movie $i$ (Defined only if $r(i, j) = 1$)
- $\theta^{(j)} :$ Parameter vector for user $j$ $(\theta^{(j)} \in \mathbb{R}^{n+1})$
- $x^{(i)} :$ Feature vector for movie $i$
- For movie $i$ and user $j$, predicted rating : $(\theta^{(j)})^T(x^{(i)})$
- $m^{(j)} :$ Number of movies rated by user $j$

## Problem Formulation

### Optimization Algorithm

#### Solving every $\theta^{(j)}$ using Linear Regression

$$
    \min_{\theta^{(j)}} \quad \frac{1}{2m^{(j)}} \sum_{i \ : \ r(i, j)=1}
    \left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right)^2 \ +
    \ \frac{\lambda}{2m^{(j)}} \sum_{k=1}^n \left( \theta_k^{(j)} \right)^2
$$

#### But $m_j$ can be treated as a constant and removed
$$
    \min_{\theta^{(j)}} \quad \frac{1}{2} \sum_{i \ : \ r(i, j)=1}
    \left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right)^2 \ +
    \ \frac{\lambda}{2} \sum_{k=1}^n \left( \theta_k^{(j)} \right)^2
$$

#### Solving all the thetas at once, 
$$
    J(\theta^{(1)}, \theta^{(2)} \dots \theta^{(n_u)}) = 
    \min_{\theta^{(1)}, \theta^{(2)} \dots \theta^{(n_u)}}
    \quad \frac{1}{2} \sum_{j=1}^{n_u} \ \sum_{i \ : \ r(i, j)=1}
    \left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right)^2 \ +
    \ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n 
    \left( \theta_k^{(j)} \right)^2
$$

#### Gradient Descent
$$
    \begin{align*}
	    \theta_k^{(j)} :=  \quad& \theta_k^{(j)} - \alpha \left( \sum_{i \ : \ r(i, j)=1}
            \left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right) x_k^{(i)} \right) 
            \quad (\text{for k = 0}) \\
       \theta_k^{(j)} :=  \quad& \theta_k^{(j)} - \alpha \left( \sum_{i \ : \ r(i, j)=1}
            \left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right) x_k^{(i)} \ + 
            \ \lambda \theta_k^{(j)} \right) 
            \quad (\text{for k > 0}) \\
    \end{align*}
$$


## Collaborative Filtering
- Feature Learning
- We construct the dual of the previous problem 
- Take some random set of values of theta, solve x, solve theta using x, solve x using theta and so on. But there is a more efficient way to do it explained in Collaborative Filering Optimization Objective

### Optimization Objective

#### Given $\theta^{(j)}, \theta^{(2)} \dots \theta^{(n_u)}$ to learn $x^{(i)}$
$$
    \min_{x^{(i)}} \quad \frac{1}{2} \sum_{j \ : \ r(i, j)=1}
    \left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right)^2 \ +
    \ \frac{\lambda}{2} \sum_{k=1}^n \left( x_k^{(j)} \right)^2
$$

#### Given $\theta^{(1)}, \theta^{(2)} \dots \theta^{(n_u)}$ to learn all $x^{(i)}$
$$
	J(x^{(1)}, x^{(2)} \dots x^{(n_m)}) = 
    \min_{x^{(1)}, x^{(2)} \dots x^{(n_m)}}
    \quad \frac{1}{2} \sum_{i=1}^{n_m} \ \sum_{j \ : \ r(i, j)=1}
    \left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right)^2 \ +
    \ \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n 
    \left( x_k^{(j)} \right)^2
$$

### Collaborative Filtering Optimization Objective

#### Given $x^{(1)}, x^{(2)} \dots x^{(n_m)}$, estimate  $\theta^{(1)}, \theta^{(2)} \dots \theta^{(n_u)}$:
$$
    J(\theta^{(1)}, \theta^{(2)} \dots \theta^{(n_u)}) = 
    \min_{\theta^{(1)}, \theta^{(2)} \dots \theta^{(n_u)}}
    \quad \frac{1}{2} \sum_{j=1}^{n_u} \ \sum_{i \ : \ r(i, j)=1}
    \left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right)^2 \ +
    \ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n 
    \left( \theta_k^{(j)} \right)^2
$$

#### Given $\theta^{(1)}, \theta^{(2)} \dots \theta^{(n_u)}$, estimate $x^{(1)}, x^{(2)} \dots x^{(n_m)}$:
$$
	J(x^{(1)}, x^{(2)} \dots x^{(n_m)}) = 
    \min_{x^{(1)}, x^{(2)} \dots x^{(n_m)}}
    \quad \frac{1}{2} \sum_{i=1}^{n_m} \ \sum_{j \ : \ r(i, j)=1}
    \left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right)^2 \ +
    \ \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n 
    \left( x_k^{(j)} \right)^2
$$

#### Minimizing $x^{(1)}, x^{(2)} \dots x^{(n_m)}$ and $\theta^{(1)}, \theta^{(2)} \dots \theta^{(n_u)}$ simultaneously:
$$
	J(x^{(1)}, x^{(2)} \dots x^{(n_m)},
	\theta^{(1)}, \theta^{(2)} \dots \theta^{(n_u)}) = 
	\quad \frac{1}{2} \ \sum_{(i, j) \ : \ r(i, j)=1}
	\left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right)^2 \ + 
	\ \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n \left( x_k^{(j)} \right)^2 + 
	\ \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n  \left( \theta_k^{(j)} \right)^2
$$


### Collaborative Filtering Algorithm
1. Initialize $x^{(1)}, x^{(2)} \dots x^{(n_m)}, \theta^{(1)}, \theta^{(2)} \dots \theta^{(n_u)}$ to small random values.
2. Use Gradient descent to minimize $J$ (Or any Optimization Algorithm).
    $$
    \begin{align*}
	    x^{(i)} :=  \quad & x^{(i)} - \alpha \left( \sum_{j \ : \ r(i, j)=1}
            \left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right) \theta_k^{(j)} \ +
            \ \lambda x^{(i)} \right) \\
       \theta_k^{(j)} :=  \quad & \theta_k^{(j)} - \alpha \left( \sum_{i \ : \ r(i, j)=1}
            \left( (\theta^{(j)})^T(x^{(i)}) \ - \ y^{(i, j)} \right) x_k^{(i)} \ + 
            \ \lambda \theta_k^{(j)} \right) \\
    \end{align*}
    $$
3. For a user with parameters $\theta$ and a movie with (learned) features $x$, predict a star rating of $\theta^Tx$


## Vectorization: Low rank Matrix Factorization

$$ Y = 
\begin{bmatrix}
	(\theta^{(1)})^T(x^{(1)}) & (\theta^{(2)})^T(x^{(1)}) & \dots & (\theta^{(n_u)})^T(x^{(1)}) \\
	(\theta^{(1)})^T(x^{(2)}) & (\theta^{(2)})^T(x^{(2)}) & \dots & (\theta^{(n_u)})^T(x^{(2)}) \\
	\vdots & \vdots & \ddots & \vdots \\
	(\theta^{(1)})^T(x^{(m)}) & (\theta^{(2)})^T(x^{(m)}) & \dots & (\theta^{(n_u)})^T(x^{(m)}) \\
\end{bmatrix}
$$

Now, define 
$$ 
X \ =
	\ \begin{bmatrix}
	- & (x^{(1)})^T & - \\
	- & (x^{(2)})^T & - \\
	- & \vdots & - \\
	- & (x^{(m)})^T & - \\
	\end{bmatrix} 
	\quad \quad
\Theta \ = 
	\ \begin{bmatrix}
	- & (\theta^{(1)})^T & - \\
	- & (\theta^{(2)})^T & - \\
	- & \vdots & - \\
	- & (\theta^{(m)})^T & - \\
	\end{bmatrix}
$$

We can observe that $X$ and $\Theta$ are low rank matrices. This means $Y$ can be factorized into 2 low rank matrices since $Y = \Theta X^T$.

### Finding Related Movies
$|| x^{(i_1)} - x^{(i_2)} ||$ is small indicates that the movies are similar

## Mean Normalization
$$
Y \ = \begin{bmatrix}
    5 & 5 & 0 & 0 & ? \\
    5 & ? & ? & 0 & ? \\
    ? & 4 & 0 & ? & ? \\
    0 & 0 & 5 & 4 & ? \\
    0 & 0 & 5 & 0 & ?
\end{bmatrix}
\quad
\mu \ = \begin{bmatrix}
    2.5 \\
    2.5 \\
    2 \\
    2.25 \\
    1.25
\end{bmatrix} 
\quad \to \quad
Y \ = \begin{bmatrix}
    2.5 & 2.5 & -2.5 & -2.5 & ? \\
    2.5 & ? & ? & -2.5 & ? \\
    ? & 2 & -2 & ? & ? \\
    -2.25 & -2.25 & 2.75 & 1.75 & ? \\
    -1.25 & -1.25 & 3.75 & -1.25 & ?
\end{bmatrix}
$$

- Mean normalization about the movie
- Perform Collaborative Filtering now
- Results will be slightly better because of mean normalization
- Taking care of user that has not done a single rating might be more important than a movie not having gotten a single rating
- For user $j$, on movie $i$, predict $(\theta^{(j)})^T(x^{(i)}) + \mu_i$

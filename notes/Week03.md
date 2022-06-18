
# Classification and Representation

## Classification
* Yes or no type of Variables
* Hence, applying Linear Regression is not of much help

### Logistic Regression Model
We need $0 \leq h_\theta(x) \leq 1$
$$\text{Define } g(z) = \frac{1}{1 + e^{-z}} \quad \text{Sigmoid Function}$$
$h_\theta(x) = g(\theta^Tx)$

#### Interpretation of $h_\theta(x)$
Probability that y = 1, given x parameterized by $\theta$
$h_\theta(x) = P(y=1 | x \ ; \ \theta)$

### Decesion Boundary
Suppose our model predicts
$y = 1$ if $h_\theta(x) \geq 0.5 \implies \theta^T x \geq 0$ 
$y = 0$ if $h_\theta(x) < 0.5 \implies \theta^T x < 0$

$\theta^T x = 0 \implies h_\theta(x) = 0.5$ is where our outputs differ
and is the "surface" that seperates the positive and negative. 
Hence, this boundary is called the **decesion boundary**

For complex functions, we take more terms in the polynomial, like the square terms


## Logistic Regression Model

The cost function used for Linear Regression,
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^m \frac{1}{2}(h_\theta(x^{(i)}) - y^{(i)})^2
$$ if applied here, **will not be Convex**

And therefore, we use a different cost function

### Logistic Regression Cost Function
$$
\text{Cost}(h_\theta(x), y) = 
\begin{cases}
	-log(h_\theta(x)) & \text{if } y = 1 \\
	-log(1 - h_\theta(x)) & \text{if } y = 0 \\
\end{cases}
$$

#### Reasons for choosing the cost function
* $\text{Cost}(h_\theta(x), y) = 0 \text{ if } h_\theta(x) = y$
* $\text{Cost}(h_\theta(x), y) \to \infty \text{ if } y=0 \text{ and } h_\theta(x) \to 1$
* $\text{Cost}(h_\theta(x), y) \to \infty \text{ if } y=1 \text{ and } h_\theta(x) \to 0$

#### Compressed Cost Function
$$\text{Cost}(h_\theta(x), y) =
-ylog(h_\theta(x)) - (1-y)log(1 - h_\theta(x))$$
This compressed cost function is **Convex**

### Variables
$$
X_{m \times (n+1)} = 
\begin{bmatrix}
	- & (x^{(1)})^T & - \\
	- & (x^{(2)})^T & - \\
	- & \vdots & - \\
	- & (x^{(m)})^T & - \\
\end{bmatrix} 
\quad \quad
y_{m \times 1} = 
\begin{bmatrix}
	(y^{(1)}) \\
	(y^{(2)}) \\
	\vdots \\
	(y^{(m)}) \\
\end{bmatrix} 
$$

$$
\text{We need to find an optimal } \theta \text{ such that} \ \ 
X\theta = y \text{ holds with low error}
$$

### Overall Cost

$$\begin{align*}
	J(\theta)
	&= \frac{1}{m} \sum_{i=1}^m
		\text{Cost}(h_\theta(x^{(i)}), y^{(i)}) \\
	&= \frac{1}{m} \sum_{i=1}^m
		-ylog(h_\theta(x)) - (1-y)log(1 - h_\theta(x))
\end{align*}$$

To fit parameters $\theta$ : $\underset{\theta} {\min} \text{ } J(\theta)$
Make a prediction given new $x$: $h_\theta(x) = \frac{1}{1-e^{x^T\theta}}$


### Gradient Desent
Use Gradient Desent and repeatedly update $\theta$

We want to minimize $J(\theta)$ :

Repeat $\{$
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$ 
$\}$ (Simultaneously update all $\theta_j$)

Equivalent to 
Repeat $\{$ $$\theta_j := \theta_j - \alpha 
\frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$ $\}$ (Simultaneously update all $\theta_j$

Vectorically,
$$
\theta := \theta - \alpha 
\frac{1}{m} \sum_{i=1}^m [(h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}]
$$


```matlab
function [J, grad] = costFunction(theta, X, y)
	h = sigmoid(X * theta);
	J = mean(-y.*log(h) - (1-y).*log(1-h));
	grad = mean((h - y) .* X, axis=1);
end
```

## Advanced Optimization
```matlab
function [jVal, gradient] = costFunction(theta)
	jVal = 0;
	gradient = zeros(size(theta));
	
	% jVal = [...code to compute J(theta)...];
	% gradient = [...code to compute derivative of J(theta)...];
	% gradient returned will be a vector of dimension (n+1) 
end
```

Then we can use octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()".

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
[optTheta, functionVal, exitFlag] = ...
	fminunc(@costFunction, initialTheta, options);
```

## Multiclass Classification: One-vs-all
Here, we approach the classification with more than two categories. 

Let's say $y$ instead of taking values from $\{0, 1\}$ can take values from $\{0, 1, \dots , n\}$

We define $n+1$ different boundaries with $y=i$ on one side and the rest on the other side. 

$y \in \{0, 1, \dots, n \}$

$h_\theta^{(0)}(x) = P(y=0|x; \theta)$
$h_\theta^{(1)}(x) = P(y=1|x; \theta)$
$\dots$
$h_\theta^{(n)}(x) = P(y=n|x; \theta)$

$\text{prediction} = \underset{i}{max} (h_\theta^{(i)}(x))$


# Solving the problem of Overfitting
When we use a lot of parameters to determine the structure of data, we might run into the case when the curve tries too hard to fit the sample data and will not be able to predict the future data that well. This is called **overfitting**

### Some ways to address overfitting

#### 1. Reduce the number of features:
-   Manually select which features to keep
-   Use a model selection algorithm
    

#### 2. Regularization
-   Keep all the features, but reduce the magnitude of parameters $\theta_j​$
-   Regularization works well when we have a lot of slightly useful features

## Cost Function

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

Suppose we want to make the below function more quadratic
$\theta_0​+\theta_1​x+\theta_2 ​x^2+\theta_3 ​x^3+\theta_4​ x^4$

We'll want to eliminate the influence of $\theta_3x^3$ and $\theta_4x^4$. Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our **cost function**:
$$
\underset{\theta}{\min} \frac{1}{2m} \sum_{i=1}^m
(h_\theta(x^{(i)}) - y^{(i)})^2 + 
1000 \cdot \theta_3^2 +
1000 \cdot \theta_4^2
$$
	
We could also regularize all of our theta parameters in a single summation as:
$$
\underset{\theta}{\min}\ \frac{1}{2m} \sum_{i=1}^m
(h_\theta(x^{(i)}) - y^{(i)})^2 + 
\lambda \sum_{j=1}^n \theta_j^2
$$

## Regularized Linear Regression
$$\begin{align*} 
	& \text{Repeat}\ \lbrace \newline 
	& \quad
		\theta_0 := \theta_0 -
		 \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \\ 
	& \quad
		\theta_j := \theta_j - \alpha\ 
		\left[ \left( \frac{1}{m}\ 
		\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + 
		\frac{\lambda}{m}\theta_j \right] 
	& \quad j \in \lbrace 1, \ 2, ... \ ,n \rbrace \newline 
	& \rbrace 
\end{align*}$$

## Regularized Logistic Regression

### Cost Function
$$
 J(\theta) = \frac{1}{m} \sum_{i=1}^m \left[
 -y^{(i)} log(h_\theta(x^{(i)} ) - 
 (1-y^{(i)} ) log(1 - h_\theta(x^{(i)} )) \right] +
 \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2
$$

$$
\frac{\partial J(\theta)} {\partial \theta_j} =
\begin{align*}
	\quad & \frac{1}{m} \sum_{i=1}^m 
	(h_\theta (x^{(i)}) - y^{(i)}) \ x^{(i)}_j
	& \quad \quad \text{for} \ j = 0 \\ 
	\quad &  \frac{1}{m} \sum_{i=1}^m 
	(h_\theta (x^{(i)}) - y^{(i)}) \ x^{(i)}_j  
	\ + \ \frac{\lambda}{m} \theta_j
	& \quad \quad \text{for} \ j > 0
\end{align*}
$$

```matlab
function [J, grad] = costFunctionReg(Theta, X, y, lambda)
	m = length(y);  % Number of training examples
	h = sigmoid(X * Theta);  % Here, X is X', and hence everything is
	J = mean(-y.*log(h) - (1-y).*log(1-h)) + ...
		(lambda/(2*m)) * sumsq(Theta(2:end));
	grad = mean((h - y) .* X, axis=1) + ...
		((lambda/m) * [0; Theta(2:end)])';
end
```


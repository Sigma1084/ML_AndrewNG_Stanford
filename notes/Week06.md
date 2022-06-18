# Evaluating a Hypothesis

-   Getting more training examples
-   Trying smaller sets of features
-   Trying additional feature
-   Trying polynomial features
-   Increasing or decreasing $\lambda$


## Evaluation Procedure
Split up the data into two sets, a **training set** and a **test set**. Typically, the training set consists of **70%** of your data and the test set is the remaining **30%**.

### Steps

1. Learn $\Theta$ and minimize $J_{train}(\Theta)$ using the **training set**
2. Compute $J_{test}(\Theta)$

### Average Test Set Error $\ \mathbf{J_{test}(\Theta)}$
1. Linear Regression 
	$$
	J_{test}(\Theta) = \frac{1}{2m_{test}} \sum_{i=1}^{m_{test}}
	\left( h_\Theta (x_{test}^{(i)}) - y_{test}^{(i)} \right)^2
	$$
2. Classification
	$$
	err(h_\Theta (x), \ y) = 
	\begin{cases}
		1 & \quad \text{if } \ h_\Theta(x) \geq 0.5 \text{ and } y=0 \\
		1 & \quad \text{if } \ h_\Theta(x) \leq 0.5 \text{ and } y=1 \\
		0 & \quad \text{otherwise} \\
	\end{cases} \\
	J_{test}(\Theta) = \frac{1}{m_{test}} \sum_{i=1}^{m_{test}}
	err \left( h_\Theta (x_{test}^{(i)}) , \ y_{test}^{(i)} \right)
	$$


## Validation

Given many models with **different polynomial degrees**, we can use a systematic approach to identify the **'best'** function. 

In order to choose the model of your hypothesis, you can test each degree of polynomial and look at the error result.

One way to break down our dataset into the three sets is:
-   Training set: 60%
-   Cross validation set: 20%
-   Test set: 20%

### Calculating Errors

1.  Optimize $\Theta$ using the training set for each polynomial degree.

2.  Find the polynomial degree **d** with the **least error** using the cross validation set.

3.  Estimate the generalization error using the test set with $J_{test}(\Theta^{(d)})$


## Diagonizing Bias vs. Variance

We examine the relationship between the **degree of the polynomial d** and the **underfitting or overfitting** of our hypothesis.

The training error will tend to **decrease** as we increase the degree d of the polynomial.

At the same time, the cross validation error will tend to **decrease** as we increase d up to a point, and then it will **increase** as d is increased, forming a convex curve.

###  Problems
- **High bias (underfitting)**: both $J_{train}(\Theta)$ and $J_{CV}(\Theta)$ will be high. 
	Also, $J_{CV}(\Theta) \approx J_{train}(\Theta)$

- **High variance (overfitting)**: $J_{train}(\Theta)$ will be low and $J_{CV}(\Theta)$  will be much greater than $J_{train}(\Theta)$


## Regularization and Bias/Variance

By observation, we can see that 

- As $\lambda$ increases, the fit is rigid (underfit)
- As $\lambda$ decreases, we tend to overfit the model

To find the $\lambda$ that is just right,
1.  Create a list of $\lambda$'s , $\left( \text{i.e. } \lambda \in \{ 0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24 \} \right)$
    
2.  Create a set of models with different degrees or any other variants

3.  Iterate through the $\lambda$'s and for each $\lambda$ go through all the models to learn some $\Theta$

4.  Select the best combo that produces the lowest error on the cross validation set.

5.  Using the best combo $\Theta$ and $\lambda$, apply it on $J_{test}(\Theta)$ to see if it has a good generalization of the problem.

## Learning Curves

![high_bias](https://drive.google.com/uc?id=1KgRnFSXOMtx4XsyUbAnrwzoB2i3-SMvX)
![high_variance](https://drive.google.com/uc?id=1qrg5_H552I8lfTlX09jZkwj5gTaBtaLi)



## Building a Spam Classifier

### Abstract

Building a ML model that takes in an email as the input and determines whether the email is a spam

### Features ( $x$ )
- Choose 100 words indicative of spam
(Most frequently occuring n words in training set)
 - Develop sophisticated  features based on the routing techneque
 - Develop sophisticated algorithms to detech misspellings (e.g. m0rtgage)

### Recommended Approach (Error Analysis)

-   Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.
-   Plot learning curves to decide if more data, more features, etc. are likely to help.
-   Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.


## Error Metrics for Skewed Class

Suppose we're training a Logistic Regression Model $h_\theta(x)$.
( $y = 1$ if cancer, $y = 0$ otherwise )
Find you got 1% error on Test Set (99% correct).

Only 99.5% of the patients have cancer.
```matlab
function y = predictCancer(x)
	y = 0;  % Ignoring x
end
```
The above function gives 99.5% accuracy but we can see that accuracy is not the right form of how correct the algorithm is for Skewed Classes.

### Precision / Recall

$y=1$ in presence of rare class that we want to detect

|                 |  **Actual 1**  |  **Actual 0**  |
|-----------------|----------------|----------------|
| **Predicted 1** | True Positive  | False Positive |
| **Predicted 0** | False Negative | True Negative  |



#### Precision
Of all the patients who were predicted $y=1$, how many actually had cancer

$$
\text{Precision} = \frac{\text{True Positives}}{\text{Predicted Positives}}
= \frac{\text{True Pos}}{\text{True Pos + False Pos}}
$$ 

#### Recall
Of all the patients who actually had cancer, what fraction were detected

$$
\text{Recall} = \frac{\text{True Positives}}{\text{Actual Positives}}
= \frac{\text{True Pos}}{\text{True Pos + False Neg}}
$$


### Trading off precision and recall

It might get pretty challenging to determine which combination of $P$ and $R$ is better and average is not quite a good way yo determine since we don't want either $P$ or $R$ to be close to 0
$F_1$ score is one of the good metric which is used my people in ML to determine which combination of $P$ and $R$ is better

#### $F_1$ score
$$
F_1 \text{ Score} := 2\frac{PR}{P+R} 
$$


## Using Large Data Sets

Assume feature $x \in \mathbb{R}^{n+1}$ has **sufficient information** to predict $y$ accurately.

### Large Data Rationale
- Using a learning algorithm with a many parameters will result in **low bias**
- Using a large data set will result in **low variance**



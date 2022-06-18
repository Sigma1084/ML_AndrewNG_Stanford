# Support Vector Machine (SVM)

## Alternate view of Logical Regression

$$
h_\theta(x) = \frac{1}{1 + e^{-\theta^Tx}}
$$
- If $y=1$, we need $h_\theta(x) \approx 1 ,\quad \theta^Tx \gg 0$
- If $y=0$, we need $h_\theta(x) \approx 0 ,\quad \theta^Tx \gg 0$

### Idea
$$
\min_\theta \frac{1}{m} 
\sum_{i=1}^{m} \left[
y^{(i)} \left(-\log h_\theta(x^{(i)}) \right)
\ + \ 
(1 - y^{(i)}) \left( -\log(1 - h_\theta(x^{(i)})) \right)
\right] \ + \ 
\frac{\lambda}{2m} 
\sum_{j=1}^n \theta_j^2
$$

### SVM Objective
$$
\min_\theta C $$
\min_\theta \frac{1}{m} 
\sum_{i=1}^{m} \left[
y^{(i)} \left(-\log h_\theta(x^{(i)}) \right)
\ + \ 
(1 - y^{(i)}) \left( -\log(1 - h_\theta(x^{(i)})) \right)
\right] \ + \ 
\frac{\lambda}{2m} 
\sum_{j=1}^n \theta_j^2
$$
\sum_{i=1}^{m} \left[
y^{(i)} \text{cost}_1(\theta^Tx^{(i)})
\ + \ 
(1 - y^{(i)}) \text{cost}_0(\theta^Tx^{(i)})
\right] \ + \ 
\frac{1}{2} 
\sum_{j=1}^n \theta_j^2
$$


### SVM Constraints
- If $y=1$, we need  $\theta^Tx \geq 1$ (not just $\geq 0$)
- If $y=1$, we need  $\theta^Tx \leq -1$ (not just $< 0$)


## Kernel

### Similarity
$$
f = \ \text{similarity}(x, l) := \exp (- \frac{\vert\vert x - l \vert\vert^2}{2\sigma^2})
$$
$x \to l \implies f \to 1$
$||x - l|| \to \infty \implies f \to 0$

### SVMs with Kernels
Given $(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ... , (x^{(m)}, y^{(m)}).$
choose $l^{(1)} = x^{(1)}, l^{(2)} = x^{(2)}, ... , l^{(m)} = x^{(m)}.$

Given example $x:$
$$
f_1 = \text{similarity}(x, l^{(1)}) \\
f_2 = \text{similarity}(x, l^{(2)})
$$

<br>

For training example $(x^{(i)}, y^{(i)})$,
$$
x^{(i)} \ \to \quad
\begin{align*} 
	f_1^{(i)} & = & \text{similarity}(x^{(1)}, \ l^{(1)}) \\ 
	f_2^{(i)} & = & \text{similarity}(x^{(2)}, \ l^{(2)}) \\
	\vdots \\
	f_m^{(i)} & = & \text{similarity}(x^{(m)}, \ l^{(m)}) \\
\end{align*}
$$

### Hypothesis
Given $x,$ compute features $f \in \mathbb{R}^{m+1}$
Predict $\ y=1 \quad \text{ if } \quad \theta^Tf \geq 0$

### Training
$$
\sum_{i=1}^{m} \left[
y^{(i)} \text{cost}_1(\theta^Tx^{(i)})
\ + \ 
(1 - y^{(i)}) \text{cost}_0(\theta^Tx^{(i)})
\right] \ + \ 
\frac{1}{2} 
\sum_{j=1}^m \theta_j^2
$$


## SVM Parameters

### $\text{C} \ ( \ = \frac{1}{\lambda})$
- Large C: Lower bias, higher variance
- Small C: High bias, low variance

### $\sigma^2$
- Large $\sigma^2$: Features vary more smoothly. Higher bias, low variance
- Small $\sigma^2$: Features vary less smoothly. Lower bias, high variance

### Similarity Function
```matlab
function  sim = gaussianKernel(x1, x2, sigma)
	sim = exp(- norm(x1-x2)^2 / (2*sigma*sigma));
end
```
Note: Peform scaling before using Gaussian Kernel

## Other choices of Kernel
Not all similarity functions are kernels
(Need to satusfy "Mercer's Theorem" to make  sure SVM's packages optimizations run correctly and do not diverge)
Some other kernels are 
- Polynomial Kernel 
- String Kernel
- Chi-square kernel
- histogram intersection kernel


## Multiclass Classification
Many SVM packages already have built in multi-class classification functionality.
Otherwise, use one-vs-all method. (Train $k$ SVMs, one to distinguish $y=i$ from the rest, for $i=1, 2, \dots k$, get $\theta^{(1)}, \theta^{(2)},\dots, \theta^{(k)}$
Pick class $i$ with largest $(\theta^{(i)})^Tx$

## Logistic Regression vs SVMs
- $n$ : Number of features ($x \in \mathbb{R}^{n+1}$)
- $m$ : Number of training examples 

<hr>

- Case 1: When $n$ is very large (relative to $m$)  $n \geq m$
	-> Use Logistic Regression or SVM without a kernel (Linear Kernel)
 
- Case 2: When n is small and m is imtermediate
	-> Use SVM with Gaussian Kernel

- Case 3: When n is small and m is large
	-> Create/Add more features and use logistic regression or SVM without a kernel.

Note: A Neaural Network can be used for any of these but is slow and harder to train

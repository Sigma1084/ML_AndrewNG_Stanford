# Neural Networks

## Cost Function and Gradient
- $L$ : Total number of layers in the network
- $s_l$ : Number of units (not counting bias unit) in layer l
- $K$ : Number of output units/classes
- $h_\Theta ​(x)_k$ : Hypothesis resulting in the $k^{th}$ output	

#### Logostic Regression Cost Function
$$
J(\theta) = - \frac{1}{m} 
\sum_{i=1}^{m} \left[ 
-y^{(i)} log(h_\theta(x^{(i)} ) - 
 (1-y^{(i)} ) log(1 - h_\theta(x^{(i)} )) 
\right] + 
\frac{\lambda}{2m} 
\sum_{j=1}^{n} \theta_j^2 \ 
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

### Cost Function for Neural Networks
$\text{Note}: \ K \text{ is the number of nodes in the final layer}$
$$
\begin{gather*} 
	J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K 
	\left[ y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + 
	(1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)
	\right] + \frac{\lambda}{2m} \sum_{l=1}^{L-1} 
	\sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2
\end{gather*}
$$

$$
\frac{\partial J(\Theta)}{\partial \Theta^{(l)}_{ij}} \text{ is calculated using Backpropagation for all layers} 
$$

## Backpropagation 
We need $\ \underset{\Theta}{\min} \ J(\Theta)$
Need to Compute
- $J(\Theta)$
- $\frac{\partial}{\partial \Theta^{(l)}_{ij}} J(\Theta)$

<br>

Suppose we have a Neural Network having 4 layers having $s_1$, $s_2$, $s_3$, $s_4$ elements with $\Theta^{(1)}$, $\Theta^{(2)}$, $\Theta^{(3)}$

Given a training set $(X, y)$, we first propagate forward
$a^{(1)} = [\text{ones} \ ; \ X], \text{ (prepend ones)} \ ((s_1+1) \times m)$
$z^{(2)} = \Theta^{(1)} a^{(1)}$
$a^{(2)} = [\text{ones} \ ; \ g(z^{(2)})], \text{ (prepend ones)} \ ((s_2+1) \times m)$
$z^{(3)} = \Theta^{(2)} a^{(2)}$
$a^{(3)} = [\text{ones} \ ; \ g(z^{(3)})], \text{ (prepend ones)} \ ((s_3+1) \times m)$
$z^{(4)} = \Theta^{(3)} a^{(3)}$
$a^{(4)} = [\text{ones} \ ; \ g(z^{(4)})], \text{ (prepend ones)} \ ((s_4+1) \times m)$

Now, the final layer
$h = a^{(4)} \text{(2:end)} \quad \text{(excluding the first element)}$

Calculating $\delta^{(4)}$, $\delta^{(3)}$, $\delta^{(2)}$
$\delta^{(4)} = h - y \quad (s_4 \times m)$
$\delta^{(3)} = (\Theta^{(3)})^T \delta^{(4)} \ .* \ g'(z^{(3)}) \quad (\text{Discard first row}) \ (s_3 \times m)$
$\delta^{(2)} = (\Theta^{(2)})^T \delta^{(3)} \ .* \ g'(z^{(2)}) \quad (\text{Discard first row}) (s_2\times m)$

Calculate $\Delta^{(3)}$, $\Delta^{(2)}$, $\Delta^{(1)}$
$\Delta^{(3)} = \delta^{(4)} * (a^{(3)})^T \ \ (s_4 \times(s_3+1))$
$\Delta^{(2)} = \delta^{(3)} * (a^{(2)})^T \ \ (s_3 \times(s_2+1))$
$\Delta^{(1)} = \delta^{(2)} * (a^{(2)})^T \ \ (s_2 \times(s_1+1))$

Finally,
$$
\frac{\partial}{\partial \Theta^{(l)}_{ij}} J(\Theta) = 
\begin{align*}
	\quad & \frac{1}{m} \Delta^{(l)}
	& \quad \quad \text{for} \ j = 0 \\ 
	\\
	\quad & \frac{1}{m} \Delta^{(l)}
	\ + \ \frac{\lambda}{m} \Theta^{(l)}_{ij}
	& \quad \quad \text{for} \ j > 0
\end{align*}
$$

## Implementation

It is not possible to store variable sized arrays or matrices in Octave or Matlab and hence, we unroll all the values into a column matrix

### Sigmoid Gradient

$$g'(x) = g(x) \ * \ (1 - g(x))$$
```matlab
function _sigGrad = sigmoidGradient(z)
	_sigGrad = sigmoid(z) .* (1 - sigmoid(z));
end
```

### Cost Function and Gradient

Illustration for a Neural Network of 3 layers  (1 hidden layer)
```matlab
function [J grad] = nnCostFunction(nn_params, ...
			s1, s2, s3, X, y, lambda)

	% Description: Calculates J and grad
	% 	X is a matrix of size (s1 * m)
	% 	y is a row matrix having the output
	% 	y==k is a matrix of size (1 * m)
	
	Theta1 = reshape(nn_params(1 : s2*(s1 + 1)), s2, s1+1);
	Theta2 = reshape(nn_params(1 + s2*(s1 + 1) : end), s3, s2+1);
	
	m = size(X, 2);  % Number of items in the dataset
	
	a1 = [ones(1, size(X, 2)); X];
	[z2, a2] = forward(Theta1, a1);
	[z3, a3] = forward(Theta2, a2);
	
	h = a3(2:end, :);  % Discarding the first row

	for  k=1:s3
		J -= mean((y==k).*log(h(k, :)) + (1-(y==k)).*log(1 - h(k, :)));
	endfor
	J += (lambda/(2*m)) * (sumsq(Theta1(:, 2:end)(:)) + ...
				sumsq(Theta2(:, 2:end)(:)));
	
	
	% Starting Backtracking now
	delta3 = h;
	for  k=1:s3
		delta3(k, :) -= (y==k);
	endfor
	delta2 = (Theta2'*delta3) .* sigmoidGradient(z2);
	
	Theta2_grad = (1/m) * delta3 * a2';
	Theta2_grad(:, 2:end) += (lambda/m) * Theta2(:, 2:end);
	Theta1_grad = (1/m) * delta2 * a1';
	Theta1_grad(:, 2:end) += (lambda/m) * Theta1(:, 2:end);

	% Unroll gradients
	grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

function [_z _a] = forward(Theta, aPrev)
	_z = Theta * aPrev;
	_a = [ones(1, size(aPrev, 2)); sigmoid(_z)];
end
```

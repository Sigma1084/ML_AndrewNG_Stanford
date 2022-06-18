# Neural Networks

We denote the parameters we have by a column matrix. 
Suppose $x^{(i)}$ be the data we have. 
Suppose there are $n$ input parameters

We denote the input data by a Matrix X
$$
X_{m \times (n+1)} ^ T = 
\begin{bmatrix}
	- & (x^{(1)})^T & - \\
	- & (x^{(2)})^T & - \\
	- & \vdots & - \\
	- & (x^{(m)})^T & - \\
\end{bmatrix}
$$

Let's say we have 1 hidden layer of size `hidden_layer_size`($s_2$) and output layer of size `num_labels`($s_3$)

Transformation from  $l_1$ to $l_2$ will need a matrix of size $s_2 \times (n+1)$, given by matrix $\ \Theta^{(1)}$

Transformation from  $l_2$ to $l_3$ will need a matrix of size $s_3 \times (s_2 + 1)$, given by matrix $\ \Theta^{(2)}$


## Forward Propagation in a NN

Consider propagating a single input vector $x$

$\text{Layer 1}$
$$
a^{(1)} = x=
\begin{bmatrix}
   1 \\ x_1 \\ \vdots \\ x_n \\
\end{bmatrix} _{(n+1) \times 1}
$$

$\text{Layer 2}$
$$
z^{(2)} = \Theta^{(1)}a^{(1)} =
\begin{bmatrix}
    \Theta^{(1)}_{10} & \Theta^{(1)}_{11} & \dots & \Theta^{(1)}_{1n}  \\
    \Theta^{(1)}_{20} & \Theta^{(1)}_{21} & \dots & \Theta^{(1)}_{2n}  \\
    \vdots & \vdots & \ddots & \vdots  \\
    \Theta^{(1)}_{s_20} & \Theta^{(1)}_{s_21} & \dots & \Theta^{(1)}_{s_2n}  \\
\end{bmatrix}
\begin{bmatrix}
    1 \\ x_1 \\ x_2 \\x_3 \\ \vdots \\  x_n
\end{bmatrix} =
\begin{bmatrix}
   z^{(2)}_1 \\ z^{(2)}_2 \\ \vdots \\  z^{(2)}_{s_2}
\end{bmatrix}
$$

$$
a^{(2)} =
\begin{bmatrix}
   1 \\ \text{sigmoid}(z^{(2)}_1) \\ \vdots \\ \text{sigmoid}(z^{(2)}_{s_2}) \\
\end{bmatrix} _{(s_2+1) \times 1}
$$

$\text{Layer 3}$

$z^{(3)} = \Theta^{(2)}a^{(2)}$
$a^{(3)} = [1; \ \text{sigmoid}(z^{(3)})]$

## Code

Some useful code for forward propagation of a Neural Network


### Sigmoid

$$g(x) = \frac{1}{1 + e^{-x}}$$
```matlab
function  _sigmoid = sigmoid(z)
	_sigmoid = 1.0 ./ (1.0 + exp(-z));
end
```

### Forward

$z^{(l+1)} = \Theta^{(l)}a^{(l)}$
$a^{(l+1)} = [1; \ g(z^{(l+1)})]$
$\text{Note: Dimensions of } \Theta^{(l)} \ : \ s_{l+1} \times (s_l + 1)$
```matlab
function [_z, _a] = forward(Theta, aPrev)
	_z = Theta * aPrev;
	_a = [ones(1, size(aPrev, 2)); sigmoid(_z)];
end
```

### Final Layer
Only for the final layer, we calculate $a^{(L)}$ and discard the first row (of ones)


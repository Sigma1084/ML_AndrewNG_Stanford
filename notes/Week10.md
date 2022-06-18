# Learning with Large Datasets

> **Machine Learning and Data**
> "It's not who has the best algorithm that wins. It's who has the most data."

### Note
When learning with large datasets (say m = 100,000,000), it's better to experiment with a smaller dataset (say m = 1,000) first and check the model. If the model has **high bias**, it's better to add new features, more training examples won't help. 

## Linear Regression Cost Function 

$$
\begin{align*}
	h_\theta(x) = & \sum_{j=0}^n \theta_jx_j \\
	J_\text{train}(\theta) = & \frac{1}{2m}
		\sum_{i=1}^m \left( h_\theta(x^{(i)}) \ - \ y^{(i)} \right)^2
\end{align*}
$$


## Gradient descent (Batch Gradient Descent)


$\text{Repeat } \{$
$$
\theta_j := \theta_j \ - \ \alpha \frac{1}{m}
\sum_{i=1}^m \left( h_\theta(x^{(i)}) \ - \ y^{(i)} 
\right) x_j^{(i)} \quad \quad 
\left( \forall \ j \in[0, n] \right)
$$ $\}$

- **Each iteration**: Uses $m$ examples $O(m)$ ; **Overall**: $O(m^2)$
-  (Batch Graident Descent with batch size = m)



## Stochastic Gradient Descent

$\text{Randomly shuffle (reorder) training examples}$
$\text{Repeat (around 1-10 times) } \{$ 
$\quad \text{for } i := 1, 2, \dots, m \ \{$ 
$\quad \quad \theta_j := \theta_j \ - \ \alpha 
\left( h_\theta(x^{(i)}) \ - \ y^{(i)} 
\right) x_j^{(i)} \quad \quad 
\left( \forall \ j \in[0, n] \right)$ 
$\quad \}$
$\}$

- **Each iteration**: Uses $1$ example $O(1)$ ; **Overall**: $O(m)$
- In $i^{th}$ iteration, tries to fit only the $i^{th}$ training example better. 
-  Shuffling and repeating multiple times helps
- Generally oscillates around the local minima without converging


## Mini Batch Gradient Descent

$\text{Say for bacth size } \ b=10, \ \text{training set size } \ m=1000$
$\text{Repeat } \{$ 
$\quad \text{for } k = 1, 11, \dots, 991 \ \{$ 
$$
\theta_j = \theta_j \ - \ \alpha \frac{1}{10}
\sum_{i=k}^{k+9} \left( h_\theta(x^{(i)}) \ - \ y^{(i)} 
\right) x_j^{(i)} \quad \quad 
\left( \forall \ j \in[0, n] \right)
$$ $\quad \}$
$\}$

- **Each iteration**: Uses $b$ example $O(b)$ ; **Overall**: $O(bm)$


## Stochastic Gradient Descent Convergence

### Checking for Convergence

#### Batch gradient descent:
- Plot $J_\text{train}(\theta)$ as a function of number of iterations of gradient descent. 
- This is a costly step and takes $O(m)$ for every iteration
- $$
  J_\text{train}(\theta) = \frac{1}{2m} \sum_{i=1}^m 
  \left( h_\theta(x^{(i)} - y^{(i)} \right)^2
  $$

#### Stochastic Gradient Descent
- $$
  \text{cost}(\theta, (x^{(i)}, y^{(i)})) = \frac{1}{2}  
  \left( h_\theta(x^{(i)}) - y^{(i)} \right)
  $$
- During learning, compute $\text{cost} \left( \theta, (x^{(i)}, y^{(i)}) \right)$ before updating $\theta$ using $(x^{(i)}, y^{(i)})$
- Every 1000 examples (say), plot $\text{cost} \left( \theta, (x^{(i)}, y^{(i)}) \right)$ averaged over the last 1000 examples processed by the algorithm.
- Learning rate $\alpha$ is typically held constant. Can slowly decrease $\alpha$ over time if we want $\alpha$ to converge.
- Example for $\alpha$ to converge, 
  $$
  \alpha = \frac{\text{constant}_1}{\text{iterationNumber } + \text{ constant}_2}
  $$


## Online Learning

$\text{Repeat Forever } \{$
$\quad \text{Get } (x, y) \text{ corresponding to the user }$
$\quad \text{Update } \theta \text{ using } (x, y) \ \{$
$\quad \quad \theta_j := \theta_j \ - \ \alpha 
\left( h_\theta(x) \ - \ y 
\right) x_j \quad \quad 
\left( \forall \ j \in[0, n] \right)$
$\quad \}$
$\} \text{ Can adapt to changing user preference}$

### Example : Shipping Service
- Shipping service website where user comes, specifies **origin** and **destination**. 
- You offer to ship their package for some **asking price**
- Users sometimes choose to use your shipping service $(y=1)$ and sometimes they don't $(y=0)$
- Features $x$ capture properties of user, of origin/destination and asking price. 
- We want to learn $p(y=1 \ | \ x , \theta)$ to optimize price.



### Example : Product Search
- User searches for "Android phone 1080p camera"
- Have 100 phones in store. Will return 10 results.
- $x:$ features of the phone, how many words in the user query match the phone's name, description etc...
- $y=1$ if the user clicks on the link, $y=0$ otherwise
- Learn $p(y=1 \ | \ x , \theta)$
- Use to show 10 phones the user is most likely to click on.

### Other common examples
- Choosing special offers to show user
- Customized selection of news article
- Product Recommendation


# MapReduce and Data Parallelism

- Enables scaling of the current algorithms
- Reduces Computational time by multi processing

## Batch Gradient Descent 
$(\text{Say }m = 400)$
$$
\theta_j := \theta_j \ - \ \alpha \frac{1}{400}
\sum_{i=1}^{400} \left( h_\theta(x^{(i)}) \ - \ y^{(i)} 
\right) x_j^{(i)} \quad \quad 
\left( \forall \ j \in[0, n] \right)
$$

$$
\begin{align*}
	\text{Machine 1 : temp}_j^{(1)} = & \sum_{i=1}^{100} 
		\left( h_\theta(x^{(i)}) \ - \ y^{(i)} \right) x_j^{(i)} 
		\quad \quad \left( \forall \ j \in[0, n] \right) \\
	\text{Machine 2 : temp}_j^{(2)} = & \sum_{i=101}^{200} 
		\left( h_\theta(x^{(i)}) \ - \ y^{(i)} \right) x_j^{(i)}
		\quad \quad \left( \forall \ j \in[0, n] \right) \\
	\text{Machine 3 : temp}_j^{(3)} = & \sum_{i=201}^{300} 
		\left( h_\theta(x^{(i)}) \ - \ y^{(i)} \right) x_j^{(i)}
		\quad \quad \left( \forall \ j \in[0, n] \right) \\
	\text{Machine 4 : temp}_j^{(4)} = & \sum_{i=301}^{400} 
		\left( h_\theta(x^{(i)}) \ - \ y^{(i)} \right) x_j^{(i)}
		\quad \quad \left( \forall \ j \in[0, n] \right) \\
\end{align*}
$$

$\text{Finally, combine : }$

$$
\theta_j := \theta_j \ - \ \alpha \frac{1}{400}
\left( \text{temp}_j^{(1)} + \text{temp}_j^{(2)} +
\text{temp}_j^{(3)} + \text{temp}_j^{(4)} \right) 
\quad \quad \left( \forall \ j \in[0, n] \right)
$$

## General Idea
- Divide the training set into parts
- Perform operations seperately on different computers/cores
- Combining results

<br> <br>
> There are a lot of implementations of MapReduce, **Hadoop** is one such implementation

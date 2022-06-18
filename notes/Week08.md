# Unsuperwised Learning

$\text{Training Set}: \{ x^{(1)}, x^{(2)}, \dots, x^{(m)} \}$

### Appications
- Market Segmentation
- Social Network Analysis
- Organise Computer Clusters
- Astronomical Data Analysis

## K-Means Clustering
(The algorithm is an iterative algorithm)
Suppose we need to cluster a set of points into k clusters.

We randomly define k points to be centroids. 
Then, Iterate {
- Group the set into k groups, such that for each centroid, each group containing points whose closest centroid is that one
- Calculate the centroid of k groups and relocate the centroids

}

### Input
- $K \text{  (Number of clusters)}$ 
- $\text{Training Set}: \{ x^{(1)}, x^{(2)}, \dots, x^{(m)} \}$
- $x^{(i)} \in \mathbb{R}^n \text{ (Drop } x_0=1 \text{ convention})$

### K-means Optimization Objective
- $c^{(i)} :$ Index of cluster to which i^th^ element is 	currently assigned
- $\mu_k:$ Cluster centroid $k$
- $J:$ Distortion

$$
J \left( c^{(1)}, \dots, c^{(m)}, \mu_1, \dots, \mu_K \right)
= \frac{1}{m} \sum_{i=1}^m
|| x^{(i)} - \mu_{c^{(i)}} ||^2
$$
$$
\underset{\mu_1, \ \dots \ , \mu_K} \min
J \left( c^{(1)}, \dots, c^{(m)}, \mu_1, \dots, \mu_K \right)
$$

### More on K-means
- Choosing k random points to be k centroids is widely used and accepted
- The algorithm might reach a local minima and give a wrong answer. One way to counter that is to run the clustering a set number of times (say 100) and choosing the clustering with the minimal Distortion 

> **Choosing the value of k**: Elbow Method (plotting cost vs k) can be used and the value where slope gets lower can be chosen. Often times, it's hard to determine and hence no high expectations. Better to ask for what purpose we are performing k-means


## Dimensionality Reduction

### Motivation
- Data Compression
- Visualization

## PCA (Principle Compenent Analysis)

Given $m$ vectors of dimension $n$
Find $k$ vectors (in $n$ dimensions) such that the project error is minimized

### Data preprocessing
Feature Scaling / Mean Normalization
Replace each $x^{(i)}_j$ with $\frac{x^{(i)}_j - \mu_j}{\sigma_j}$

### Algorithm
Define Co-variance Matrix
$$
\Sigma = \frac{1}{m-1} \sum_{i=1}^n 
(x^{(i)}) (x^{(i)})^T
$$
Compute eigenvectors of $\Sigma$
Use SVD, since $\Sigma$ is symmetric, $U=V$
The columns of $U$ are eigen values of $\Sigma$
```matlab
[U, S, V] = svd(Sigma);
Ureduce = U(:, 1:k);
z = Ureduce' * x;
```

### Choosing k
Typically, k is chosen to be the smallest value such that 
$$
\frac
{ \frac{1}{m} \sum_{i=1}^m ||x^{(i)} - x^{(i)}_{approx}||^2 }
{ \frac{1}{m} \sum_{i=1}^m ||x^{(i)}||^2 }
\quad \leq \quad \epsilon
$$

where $\epsilon$ is usually 0.01, 0.05, 0.1

Choosing the minimum k satisfying the below constraint
$$
\frac
{ \sum_{i=1}^k S_{ii} }
{ \sum_{i=1}^m S_{ii} }
\quad \leq \quad 1-\epsilon
$$
can also be done


## Application of PCA
- Compression
	- Reduce memory/disk needed to store the data
	- Speed up the learning algorithm
- Visualization


## Bad use of PCA
- Reducing the number of features can be one of the way to **address overfitting**. But, this is a **bad use PCA**. Use **normalization** instead
- Applying PCA without checking if the model runs on original data. Simply reducing the size without application might not give the most accrate results since  after PCA, some of the data is **lost**

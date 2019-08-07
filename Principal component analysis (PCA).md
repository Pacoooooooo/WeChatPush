# 主成分分析

![1561817334684](C:\Users\PakuZhou\AppData\Roaming\Typora\typora-user-images\1561817334684.png)

之前，以为自己懂PCA的，直到看到了上面的模型，就懵圈了。。。

于是，花了些时间看了PCA的相关资料，梳理成这篇笔记。

[TOC]

## 1 从数据降维的角度

PCA就是降维：将数据从原始特征空间投影到一个新的特征空间，新的特征空间如果维数比原始空间低，那么就达到了降维的目的。这中间有许多线性代数的基础知识不会被提到，具体的内容可以阅读参考资料[5]。这部分的一些矩阵相关性质见参考资料[6]。

### 1.1 原特征空间与新空间的映射

有$M$ 个数据样本，$\boldsymbol{x}^{(m)}\in \mathbb{R}^n$ 。假定这些样本进行了中心化，再假定投影变换之后得到的新坐标系$\{\boldsymbol{w}_1,\dots,\boldsymbol{w}_k\}$ ，其中$\boldsymbol{w}_i$ 为标准正交基向量。若$k < n$，则达到降维的目的。

记$W=[\boldsymbol{w}_1,\dots,\boldsymbol{w}_k] \in \mathbb{R}^{n \times k}$ ,

原空间样本$\boldsymbol{x}^{(m)}$ 变换到新空间为 $\boldsymbol{y}^{(m)} \in \mathbb{R}^k$ ,则 $\boldsymbol{y}  = W^T \boldsymbol{x}^{(m)}$ 。

若基于$\boldsymbol{y}^{(m)}$ 来重构$\boldsymbol{x}^{(m)}$，则会得到$\tilde{\boldsymbol{x}}^{(m)} = W \boldsymbol{y}^{(m)} = W(W^T\boldsymbol{x}^{(m)})$ 。

### 1.2 从最近重构性的角度建模

假设有$M$个样本数据：$\boldsymbol{x}^{(1)}, \dots, \boldsymbol{x}^{(M)}$,它们对应的重构数据为：$\tilde{\boldsymbol{x}}^{(1)},\dots,\tilde{\boldsymbol{x}}^{(M)}$。

将最小化这两者之间的误差（用$L_2$-norm度量）作为目标：
$$
\min \sum_{m =1}^{ M} \| \boldsymbol{x}^{(m)} - \tilde{\boldsymbol{x}}^{(m)} \|_2 = \min \sum_{m =1}^{ M} \| \boldsymbol{x}^{(m)} -  W W^T \boldsymbol{x}^{(m)} \|_2
$$

$$
\begin{array}{ll}
 \quad &\sum_{m =1}^{ M} \| \boldsymbol{x}^{(m)} -  W W^T \boldsymbol{x}^{(m)} \|_2 \\
 = & \|\boldsymbol{X} - WW^T \boldsymbol{X}\|_F \\
 = & tr\Big( (\boldsymbol{X} - WW^T \boldsymbol{X})^T (\boldsymbol{X} - WW^T \boldsymbol{X} ) \Big) \\
 = & tr\Big( \boldsymbol{X}^T\boldsymbol{X} - 2 \boldsymbol{X}^T WW^T \boldsymbol{X} + \boldsymbol{X}^T WW^T WW^TX \Big) \\
 = & tr\Big( \boldsymbol{X}^T\boldsymbol{X} - 2 \boldsymbol{X}^T WW^T \boldsymbol{X} + \boldsymbol{X}^T WW^TX \Big) \;\;\;\;\;\;\;\;\;\; {\color{red}[W^TW=I]}\\
 = & tr\Big( \boldsymbol{X}^T\boldsymbol{X}  \Big) - tr\Big( \boldsymbol{X}^T WW^T \boldsymbol{X} \Big)
\end{array}
$$

所以，最近重构性的模型为：
$$
\begin{array}{ll}
&\min_{W} & - tr\Big( \boldsymbol{X}^T WW^T \boldsymbol{X} \Big)\\
&s.t. &W^TW = \boldsymbol{I}
\end{array} \tag{最小重构}
$$
使用拉格朗日乘子法，有KKT条件，可得：
$$
\boldsymbol{X}\boldsymbol{X}^TW = \lambda W
$$
只需对协方差矩阵 $\boldsymbol{X}\boldsymbol{X}^T$ 进行特征值分解，将求得的特征值排序：$\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n $ ，再去前$k$ 个特征值对应的特征向量构成 $W = (\boldsymbol{w_1},\dots,\boldsymbol{w_k})$ 。这就是主成分分析的解。

### 1.3 从最大可分性的角度建模

样本点在新空间的投影为 $W^T \boldsymbol{x}^{(m)}$ ,若所有的样本点的投影尽可能分开，则应该使投影后的样本点总方差 $\frac{1}{M}\sum_{m=1}^{M} W^T \boldsymbol{x}^{(m)} \boldsymbol{x}^{(m)T} W$ 对角元素最大，即
$$
\begin{array}{ll}
\max_W  & tr(\sum_{m=1}^{M} W^T \boldsymbol{x}^{(m)} \boldsymbol{x}^{(m)T} W)= tr(W^T\boldsymbol{X}\boldsymbol{X}^TW) \\
s.t. & W^TW = \boldsymbol{I}
\end{array}
\tag{最大可分}
$$
（**最大可分**)模型与（**最近重构**）模型等价。

## 2 从数据逼近的角度

这一部分从数据逼近的角度来得到PCA，在此过程中可以看到SVD和PCA的联系。

同时，这部分也回答了一个问题：

**为什么可以用前$k$个特征值占所有特征值的比例来衡量降维后数据的信息？**

### 2.1 Singular-Value Decomposition

If $A$ is a real $m-by-n$ matrix, then there exist orthogonal matrices $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ such that 
$$
U^T A V = \Sigma = diag(\sigma_1, \sigma_2, \dots , \sigma_p) \in \mathbb{R}^{m\times n}
$$
where $p = min\{m,n\}$ , $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_p \ge 0$ .

### 2.2 The Eckhart-Young Theorem

**The Frobenius Norm**
$$
|\boldsymbol{A}\|_{F} :=\sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} a_{i j}^{2}}=\sqrt{\operatorname{trace}\left(\boldsymbol{A}^{\top} \boldsymbol{A}\right)}
$$

$$
\|\boldsymbol{A}\|_{F}^{2}=\operatorname{trace}\left(\boldsymbol{A}^{\top} \boldsymbol{A}\right)=\operatorname{trace}\left(\left(\boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^{\top}\right)^{\top}\left(\boldsymbol{U} \boldsymbol{\Sigma} \boldsymbol{V}^{\top}\right)=\operatorname{trace}\left(\boldsymbol{\Sigma}^{\top} \boldsymbol{\Sigma}\right)=\sigma_{1}^{2}+\ldots+\sigma_{r}^{2}\right.
$$

**Low-Rank Approximation via the SVD**
$$
\boldsymbol{A}_{k} :=\sum_{i=1}^{k} \sigma_{i} \boldsymbol{u}_{i} \boldsymbol{v}_{i}^{\top}
$$

$$
\left\|\boldsymbol{A}-\boldsymbol{A}_{k}\right\|_{F}=\sqrt{\sigma_{k+1}+\ldots+\sigma_{r}}
$$

**Eckhart-Young Theorem **

If $k < r = rank(A)$ and


$$
A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T \tag{Truncate-SVD}
$$
then 
$$
\min_{rank(B)=k}  \| A - B \|_2 = \| A - A_{k} \|_2 = \sigma_{k+1}
$$

$$
\min_{rank(B)=k}  \| A - B \|_F = \| A - A_{k} \|_F = \sqrt{\sigma_{k+1}+\ldots+\sigma_{r}}
$$

The Eckhart-Young Theorem can help to detemine what value of $k$ to take in order to ensure that $X_k$ is a “sufficently good” approximation of $X$. In particular it allows to express the relative error a low-rank approximation $X_k$ in terms of the singular values of  $X$, since
$$
\frac{\left\|X-X_{k}\right\|_{F}^{2}}{\|X\|_{F}^{2}}=\frac{\sigma_{k+1}^{2}+\cdots+\sigma_{r}^{2}}{\sigma_{1}^{2}+\cdots+\sigma_{r}^{2}}
$$
<u>Thus if our goal is to ensure a given bound on the relative error (say at most 0.05), then we can find an approprate value of $k$ by examining the singular values, instead of proceeding by trial and error and computing $X_k$ for various values of $k$.</u>

### 2.3 Principal Component Analysis

In PCA, the input is a family $\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{n}$ of data points in $\mathbb{R}^m$. Write $\boldsymbol{\mu} :=\frac{1}{n} \sum_{i=1}^{n} \boldsymbol{x}_{i}$ for the mean data point. By first replacing $\boldsymbol{x}_{i}$ by $\boldsymbol{x}_{i} − \boldsymbol{\mu}$ we may assume that the input data are mean centred. 

Given a target dimension $k ≤ m$, our goal is to find points $\tilde{\boldsymbol{x}}_{1}, \ldots, \tilde{\boldsymbol{x}}_{n} \in \mathbb{R}^{m}$ such that the   **reconstruction error** $\sum_{i=1}^{n}\left\|\boldsymbol{x}_{i}-\tilde{\boldsymbol{x}}_{i}\right\|^{2}$ is minimised subject to the constraint that $\tilde{\boldsymbol{x}}_{1}, \ldots, \tilde{\boldsymbol{x}}_{n}$ lie in a subspace of $\mathbb{R}^m$ of dimension at most $k$.
$$
\boldsymbol{X} :=\left(\begin{array}{ccc}{ |} & {} & { |} \\ {\boldsymbol{x}_{1}} & {\dots} & {\boldsymbol{x}_{n}} \\ { |} & {} & { |}\end{array}\right)
$$

$$
\tilde{\boldsymbol{X}} :=\left(\begin{array}{ccc}{ |} & {}  & { |} \\ {\tilde{\boldsymbol{x}}_{1}} & {\dots} & {\tilde{\boldsymbol{x}}_{n}} \\ { |} & {} & { |}\end{array}\right)
$$

Then the reconstruction error is nothing but
$$
\|\boldsymbol{X}-\tilde{\boldsymbol{X}}\|_{F}^{2}
$$
Thus by the Eckhart-Young Theorem an optimal choice of $\tilde{\boldsymbol{X}}$ is the matrix $X_k$,

as defined in (**Truncate-SVD**).



## 3 鲁棒主成分分析

从2.3节可以看到，PCA问题可以表述成下面的问题：
$$
\begin{array}{ll}{\operatorname{minimize}} & {\|\boldsymbol{X}-\tilde{\boldsymbol{X}}\|_{F}^{2}}\\ {\text { subject to }} & {\operatorname{rank}(\tilde{\boldsymbol{X}}) \le k}\end{array}
 \tag{PCA-Model}
$$
也就是说，原始的数据矩阵 $\boldsymbol{X}$ 可以由一个低秩的矩阵 $\tilde{\boldsymbol{X}}$ 来近似，当然，近似存在着误差。

令
$$
\boldsymbol{E} = \boldsymbol{X} - \tilde{\boldsymbol{X}}
$$
当误差 $\boldsymbol{E}$ 服从高斯分布时，可以最小化 $\| \boldsymbol{E} \|_F$ ，就是上面的（**PCA-Model**）这一模型；

当误差 $\boldsymbol{E}$ 不服从高斯分布时，在现实数据中，它很可能是稀疏的，我们可以去最小化 $\| \boldsymbol{E} \|_0$：
$$
\begin{array}{ll}{\operatorname{minimize}} & {\|\boldsymbol{X}-\tilde{\boldsymbol{X}}\|_{0}}\\ {\text { subject to }} & {\operatorname{rank}(\tilde{\boldsymbol{X}}) \le k}\end{array}
$$
由于这个模型的 $L_0$-norm 和 $\operatorname{rank}(\tilde{\boldsymbol{X}})$ 都不是很好优化的问题，可以将原问题“放松”成如下的问题：
$$
\begin{array}{ll}{\operatorname{minimize}} & {\|\boldsymbol{X}-\tilde{\boldsymbol{X}}\|_{1}}\\ {\text { subject to }} & {\| \tilde{\boldsymbol{X}} \|_* 很小}\end{array}
$$
也可以写成这样：
$$
\begin{array}{ll}{\operatorname{minimize}} & {\| \boldsymbol{E} \|_{1} + \| \tilde{\boldsymbol{X}} \|_*}\\ {\text { subject to }} & {\boldsymbol{X} = \tilde{\boldsymbol{X}} + \boldsymbol{E}}\end{array} \tag{RPCA-Model}
$$
对于(**RPCA-Model**)这个模型的求解还不会，具体的可以参见参考资料[4]。





​                                                                                                                                                                                      周哲豪

​                                              																		                                                        2019.06.30

---

## 参考资料

- [1] 周志华. 《机器学习》
- [2] Gene H. Golub & Charles F. Van Loan. 《Matrix Computation》4th Edition
- [3] Principal Component Analysis https://www.cs.ox.ac.uk/people/james.worrell/SVD-thin.pdf
- [4] Emmanuel J. Cand`es, Xiaodong Li, Yi Ma, et.al. Robust Principal Component Analysis? https://statweb.stanford.edu/~candes/papers/RobustPCA.pdf
- [5] PCA的数学原理 https://zhuanlan.zhihu.com/p/21580949
- [6] Kaare Brandt Petersen &Michael Syskind Pedersen. The Matrix Cookbook http://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf



# Machine Learning Algorithms From Scratch

Implementations of several machine learning algorithms in python using numpy.

## Notebooks

The following notebooks are included:

### Supervised Learning

- [Linear Regression](linear%20regression.ipynb)
    - Simple Least Squares
    - Oridnary Least Squares
    - Bayesian Linear Regression
    - Least Mean Squares
- [Locally Weighted Regression (LWR)](LWR.ipynb)

- [Linear Classification](linear%20classification.ipynb)
    - Perceptron Learning Algorithm
    - Logistic Regression
    - Naive Bayes Classifier
    - Support Vector Machine (Not implemented from scratch)
      
- [Multilayered Perceptron (Neural Network)](https://github.com/nonkloq/nn_dqn-from-scratch/blob/main/nn-mlp_from_scratch.ipynb)

- [Multinomial & Gaussian Naive Bayes](Multinomial_and_GaussianNP.ipynb)
    - Gaussian Naive Bayes (Clone from Linear Classification)
    - Multinomial Naive Bayes



#### Inductive Learning

- [Candidate Elimination Algorithm (CEL)](CEL.ipynb)

#### Ensemble Learning

- [Decision Tree and Random Forest](trees_forest.ipynb)
    - ID3 Algorithm
    - Random Forest


### Unsupervised Learning

- [Unsupervised Learning](unsupervised%20learners.ipynb)  (contains K-means, KNN with KD tree and EM for GMM)
    - K-Means
    - K Nearest Neighbours
      - Brute Force KNN
      - KD-Tree
      - Kernels to Compute Weight: ID_weight, Epanechnikov & Tricube
    - Expectation Maximisation for Gaussian Mixture Model

- [K-Means and Expectation-Maximisation for Gaussian Mixture Model (Viz)](EM_for_GMM_and_Kmeans.ipynb) (clone from unsupervised learners)
  - Visual Comparison of K-Means vs. EM for GMM Using 2-Dimensional MNIST Data Reduced by PCA & Synthetic Data from N Gaussian Distributions.
    
- [K-Nearest Neighbors (KNN) for classification (iris dataset)](KNN_for_iris.ipynb) (clone from unsupervised learners)


## Usage

The code is provided in Jupyter Notebook format (.ipynb), which you can either view directly on GitHub or download and run on your local machine. Each notebook contains clear implementations of the algorithms, along with the relevant formulas and pseudo code for that algorithm. For better understanding of the algorithms, check out the specific notebook of interest.

## References

- Pattern Recognition and Machine Learning by Christopher Bishop
- Machine Learning: An Algorithmic Perspective, Second Edition by Stephen Marsland

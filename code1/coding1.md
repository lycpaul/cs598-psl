### kNN & LR & Bayes

- Reference: https://campuswire.com/c/GB46E5679/feed

#### k-Nearest Neighbors (k-NN)
This is a non-parametric, instance-based learning algorithm used primarily for classification (but it can also be used for regression). For classification, k-NN assigns a class label to a data point based on the majority class among its k nearest neighbors in the feature space. It’s simple and doesn’t involve training a model in the traditional sense but rather uses the training data directly to make predictions.

#### Linear Regression
This is a parametric model used for predicting a continuous outcome (regression). It assumes a linear relationship between the input features and the output. For example, it predicts a numerical value based on the input features and finds the line (or hyperplane in multiple dimensions) that best fits the data.

#### Bayes’ Rule
This is a probabilistic approach used in classification problems. It applies Bayes’ theorem to predict the probability of a class given the input features. In the context of classification, it can be used to derive models like Naive Bayes, which assumes that features are conditionally independent given the class.


-------

Reference:
- [Vectorization in numpy](https://jaykmody.com/blog/distance-matrices-with-numpy/)

-------

Below are the concepts covered from Q1 to Q11; you can use those keywords to find relevant online tutorials, wiki pages, or youtube videos. 

**Q1**: Definition of probability on random events; probability of tossing a die

**Q2**: Definition of conditional probability
$$P(A|C) = P( A \text{ and } C) / P(C)$$

**Q3**: Definition of independence of two random events. A and C are **independent** if and only if 
- $$P(A|C) = P(A)$$ (i.e., the prob of observing event A is irrelevant to whether we have observed event C or not), or equivalently
- $$P(A \text{ and }C) = P(A) \times P(C)$$ (i.e., the prob of observing both A and C is equal to the product of two individual probs)

Note that independent and mutually inclusive are different.
- **mutually exclusive**: $$P(A \text{ or } B) = P(A) + P(B)$$

- **independent** $$P(A \text{ and } B) = P(A) \times P(B)$$

**Q4**: Bayes formula

Question from Students: How to find P(B) from Bayes' theorem?
Answer: Use **Law of total probability**:
P(tested positive) = P(tested positive| without disease) * P(without disease) + P(tested positive| with disease) * P(with disease)


**Q5**: Definition of the covariance matrix

**Q6**: one-dimensional normal PDF; how to compute probabilities using PDFs. No Z-table is provided; you can use `pnorm()` function in R or use `scipy.stats.norm` in Python \[[Link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)\] to compute the result. Round your answer to two decimal places. Do not use percentages. For example, if your answer is 57.64%, input "0.58"

**Q7**: CLT; Linear combinations of normal random variables are still normally distributed

**Q8**: Definition of the trace of a matrix

**Q9**: Definition of the inner product (or dot product) between two vectors. 

Note that this question asks for the sine of the angle, not the angle itself.

**Q10**: Definition of the derivative of a one-dimensional function

**Q11**: the inverse of a 2-by-2 matrix \[[Link](https://www.mathsisfun.com/algebra/matrix-inverse.html)\]

**Q12**: Classification vs regression (covered in week 1 lectures)

**Q13**: kNN; use the Euclidian distance as the distance measure (covered in week 1 lecture, or check ISLR)

**Q14**: Supervised vs unsupervised learning (covered in week 1 lectures)

**Q15**: Solve the weights least square problem. Take the derivative and set it to zero.

$$
f(u) = \sum_i w_i (x_i - u)^2$$
$$0 = f'(u) = -\sum_i w_i 2 (x_i - u) = - 2 (\sum_i w_i x_i) + 2 u \cdot (\sum_i w_i)$$


**Q16**: Note that A and B are not necessarily independent. 
$$ P(A \cap B) = 1 - P(A^c \cup B^c) 
$$
Then use Boole's inequality $$P(A^c \cup B^c) \le P(A^c) + P(B^c)$$. 

Question from a student: a matrix A with a superscript "c"? Can you describe what this means? 
Answer: $$A$$ is an event, not a matrix.  $$A^c$$ is the complement of $$A$$. By definition, $$P(A^c) =  1 - P(A)$$.

**Q17**: Question from [slides from CS 445] #10

Note that you have unlimited attempts for quizzes until December 10. If you find this question difficult, you can always try it later. 
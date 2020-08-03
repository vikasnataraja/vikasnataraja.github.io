---
layout: post
title: Understanding Bias and Variance in Machine Learning
subtitle: A look at the bias-variance trade-off in the world of ML
cover-img: /assets/img/biasvar/cover_image.jpeg
tags: [machine-learning, bias, variance, ai-series]
---
If you've taken a machine learning class, chances are they told you about bias and variance or the bias-variance trade-offs you make in ML. Well, it turns out they are super important and in this blog I'll talk about what bias and variance are in the context of machine learning, what role they play and why they are important to consider for an ML algorithm.

The goal of any ML algorithm in general (supervised ML) is to find a function $f$ that approximates the output $Y$ for a given input data $x$. But let's start off with some basic statistical explanations of bias and variance. **Simply put, bias is simplifying assumption(s) that a model makes to approximate that function $f$ and denoted as $\hat{f}$. Variance is the amount by which that approximation or estimation of function $f$ changes when different training data or input data $X$ is fed.** Mathematically, they are characterized as sources of error that influence a prediction of output $Y$ for input $X$:

\begin{equation}
\label{eq:err}
  Err(x) = E[ (Y - \hat{f}(x))^2 ] \nonumber
\end{equation}

To break it down into the bias component and the variance component:

\begin{equation}
\label{eq:bias_var_err}
  Err(x) = (E[\hat{f}(x)] - f(x))^2 + E[(\hat{f}(x) - E[\hat{f}(x)])^2] + \sigma_{e}^2 \nonumber
\end{equation}

\begin{equation}
\label{eq:bias_var_err_part2}
  Err(x) = Bias^2 + Variance + Irreducible Error \nonumber
\end{equation}

The $sigma_{e}^2$ term is the irreducible error that characterizes the noise in the relationship between $X$ and $Y$ that cannot be reduced by any model and is out of our control. But both variance and bias matter because they determine how good a model can capture the function $f$, i.e. how good $\hat{f}$ is, that maps input data to output data. Ideally, the algorithm should minimize bias and variance so that they are both as low as possible but practically they are never zero and so it becomes a "trade-off" to balance one against the other.


## Error from Bias

Like I mentioned earlier, bias is the simplifying assumption(s) that the model makes that results in a difference between the expected prediction of the model and the actual correct value or ground truth of the data. This is obviously not ideal but you might ask - why not train the model better, make it more complex or have more data points so there are more samples to learn from? Well, reducing the bias is not as easy as you might think.

For example, let's say we had a set of points that looked like this:

<figure align="center">
  <img width="900" height="600" src="/assets/img/biasvar/data.png" alt="Dataset scatter points">
  <figcaption> Random set of data points </figcaption>
</figure>

Now, these points do not appear to be linear but let's say we did try a linear model like Linear Regression. That would look something like this:

<figure align="center">
  <img width="900" height="600" src="/assets/img/biasvar/linear_regression.png" alt="Linear regression">
  <figcaption> Linear regression model fit over the data points </figcaption>
</figure>

We can see that the model doesn't capture the essence of the data at all. The Mean Squared Error for both training and validation is very high. This is called **underfitting** which happens when the chosen model is simply not capable of modeling the data. The obvious solution might be to increase model complexity - we know the data is non-linear so let's say we chose a non-linear model to fit the data. Now, what you have to remember here is that data itself is not assumed to perfect - it will have noise and outliers. Our task is to get a model that will fit the actual data points and not the noise. If we don't do that and simply throw all the data points at once without informing the model of the noise, we might see something like this:

<figure align="center">
  <img width="900" height="600" src="/assets/img/biasvar/overfit_linear.png" alt="Overfitting">
  <figcaption> Overfitting the data points </figcaption>
</figure>

The model did too well in this case. In fact, it captured every single data point including the noisy points. Now you might think that's great because it was asked to model a set of points and it did really well. But what has actually happened is that the model has "memorized" the answers to each data point and the pattern behind them that if we introduce a new set of data points, it will fail miserably and fall apart. It is so rigid that it cannot work on anything new. In the machine learning world, you will know your model is overfitting when the training accuracy is incredibly high but the validation accuracy is significantly low because it is *biased* towards the training set. What we want is both training and validation accuracies to be somewhat similar and equal. All this is caused by bias in the model i.e. it is now biased towards a set of data points that it doesn't know how to react to a new set of points. So, whenever you see the words "memorized" or "rigid", that means there is bias in the model.


## Error from Variance

So, we know that simply increasing model complexity won't always solve the problem of underfitting because it can lead to overfitting. So what if we simply introduce more data points? Machine learning algorithms are data-hungry and naturally the more data you have, the better, right? Well, not quite. Simply adding more data does not necessarily mean we are adding good data. Very often, you will find that before feeding data to an ML algorithm, it goes through what is called pre-processing but that is a different topic altogether.

Let's say we used K-Nearest Neighbors to solve the given original dataset. That looks something like this:

<figure align="center">
  <img width="1200" height="800" src="/assets/img/biasvar/original_knn.png" alt="K-Nearest Neighbors">
  <figcaption> K-Nearest Neighbors model fit over original data points </figcaption>
</figure>

It's not too bad but it's not too great either. At first glance, it looks like the model with `n_neighbors = 3` i.e. the black line appears to do well because it has the lowest training error (MSE = Mean Squared Error) but a look at validation MSE reveals that it has clearly overfit because it is considerable higher than the training error. That leads me to make my point that when we choose a model or even train a model, we should be monitoring both the training and validation errors because just one of them being good does not guarantee the same for the other. With that in mind, we could go with `n_neighbors = 4` or `n_neighbors = 5` because they have a good balance between the two errors but `n_neighbors = 6` has too high a training error.

Now, at this stage, let's say we add more data points, about 200 additional ones. So now, our data looks like this:

<figure align="center">
  <img width="1000" height="600" src="/assets/img/biasvar/add_data.png" alt="Data points">
  <figcaption> Added data points alongside the original ones </figcaption>
</figure>


And let's say we fit this new dataset with the additional 200 data points with that same algorithm. Here's what that looks like:

<figure align="center">
  <img width="1200" height="800" src="/assets/img/biasvar/add_data_knn.png" alt="K-Nearest Neighbors">
  <figcaption> K-Nearest Neighbors model fit with additional data </figcaption>
</figure>

This definitely appears messy but stay with me for a second. Each subplot again shows a model with a different `n_neighbors` setting. The reason I am showing multiple cases of `n_neighbors` is because while all the models appear similar visually, when we take a closer look at their MSE numbers, we see that the yellow line i.e. `n_neighbors = 3` is clearly overfitting because the training and validation MSEs are so apart. But the pink, orange and purple lines i.e. `n_neighbors = 4`, `n_neighbors = 5` and `n_neighbors = 6`, do relatively well. You will notice that the trend is that as the training error increases, the validation error decreases and actually in cases where `n_neighbors = 1` and `n_neighbors = 2`, it is the opposite meaning the validation error was going up while training error was relatively low meaning it was overfitting.

The whole point of this exercise is to demonstrate that just because we add new data points doesn't mean we are necessarily helping the model work better. In fact, there is a very good chance that the same model before adding the data performs worse with the addition of new data points as we can see in the figure above. That brings me to my point about the actual trade-off we make.

## The Trade-Off

Variance and bias are both hard problems to overcome. Adding more random data points changes variance by either reducing it or increasing it and could more add non-linearity to the data itself. That means we have to make the model more complex which brings us back to dealing with bias because let's say when we added the data points, and the variance increased. We then risk the model not being able to characterize the data either due to low complexity or too high of a variance. But it does ensure that we are not biasing the model by feeding it very similar data (data having similar variances). What if we were to decrease the variance when we added the data points? Well, as you might have guessed, that isn't really solving the problem of bias but maybe we don't need a complex model anymore because the data has very low variance and can deal with it fairly easily.

Take a look at this illustration that shows the different cases for bias and variance:

<figure align="center">
  <img width="800" height="800" src="/assets/img/biasvar/biasvar_graph_illus.png" alt="Bias variance cases">
  <figcaption> Different cases of bias and variance. Source: Scott Fortmann-Roe</figcaption>
</figure>

We encounter high bias and high variance when the model we have is at the same time highly complex and completely inappropriate which means we are using something like a third order polynomial model to fit a dataset that actually has correlations in another dimension meaning we are using the wrong model for the wrong data. The flip side of that is low bias and low variance which is what a good ML algorithm must strive to achieve. Low bias means that the model is just complex enough to capture the data correlations while also having low variance which again means that the model has just enough parameters to deal with variances in the data. This is essentially the bias-variance trade-off we have to deal with.


<figure align="center">
  <img width="900" height="600" src="/assets/img/biasvar/biasvariance.png" alt="Bias variance graph">
  <figcaption> Bias and variance plotted against model complexity </figcaption>
</figure>

So, to summarize, the bias-variance trade-off is where we train a model with the right complexity and at the same time use the right data and correlations between them. High bias and low variance occurs when the model is simple enough to work on a limited set of data such as the training data but fails when it has to deal with new or unseen data from different statistical regions. Low bias and high variance occurs when the model is highly complex and has been trained on a variety of ranges of data variances but still struggles to deal with data from the same region. We also saw that adding more data cannot be an end-all be-all solution to these problems because the data itself needs to be the right data. Balancing the bias and variance is a topic of discussion for ML researchers to this day mostly because it is dependent on so many factors - the data, the correlations between features, the model complexity, parameters in the model and so on. It is a highly interesting topic because at its core, it is very basic math problem that machine learning can struggle to overcome. There are a lot of variables to look after and I hope this blog introduced you to some of the challenges that are present to this day in the ML world.

References: [Scott Fortmann-Roe's excellent blog post](http://scott.fortmann-roe.com/docs/BiasVariance.html#fn:2)


Cover Image credit: [Roman Mager](https://unsplash.com/@roman_lazygeek)

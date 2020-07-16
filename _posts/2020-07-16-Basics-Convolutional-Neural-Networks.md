---
layout: post
title: Basics of Convolutional Neural Networks
subtitle: Convolution, filters, layers and feature extraction
cover-img: /assets/img/basics_cnn/cover_image.jpeg
tags: [cnn, ai-series]
---
 Neural networks have truly transformed the way machine learning and AI in general are perceived and to an extent, researched. They have gone from the fringes of the CS community where only select academics in the late 1980s would publish a paper occasionally to becoming synonymous with artificial intelligence and being capable of solving some of the world's biggest problems, be it self-driving cars or a Mario Kart game :) Convolutional Neural Networks (or CNNs) have played an incredibly important in that turnaround and today, many starter machine learning courses start with this particular topic because of their inherent visual aspect and shorter training or learning times. But convolution? What is that? What does it do and what is its connection to neural networks? In this blog, I will attempt to answer those questions. But let's go back a little to truly appreciate the progress we have made in neural networks and deep learning.

 Let's start with Yann LeCun's pioneering work in 1989 in applying backpropagation, which later became a founding principle of neural networks, to a practical application by [recognizing handwritten digits of a zip code for the U.S. Postal Service](https://www.mitpressjournals.org/doi/10.1162/neco.1989.1.4.541). In 1998, he introduced [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), a convolutional neural network that that was the culmination of more than a decade's work which was able to recognize characters in a document such as bank checks, technology that we still use today albeit more advanced. Of course, since then we have made significant progress but this quote from that paper seems to capture the usefulness of using such technology: *"...better pattern recognition systems can be built by relying more on automatic learning, and less on hand-designed heuristics."* That might not make a lot of sense but the key concept I want to underline here is ***pattern recognition*** which is exactly what CNNs do. They recognize patterns that may be multi-dimensional and beyond human capability and spit out a prediction of what that pattern might be probabilistically (slight tangent here - it is notable that even though there was progress after LeCun's '98 paper, CNNs didn't really become popular until 2012 with the advent of GPUs and other hardware accelerators to go with Alex Krizhevsky's AlexNet CNN. This is very interesting and at the same time confusing to me and perhaps I will dive into the history a bit more in another blog :)).

 Now, I still haven't explained what a convolution is, have I? Well, if you think back to Calculus-1 (or just look it up on wikipedia), you'll probably remember an equation that looked like this:

 \begin{equation}
 \label{eq:conv_function_cont}
   (f * g) [t] \triangleq \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau \nonumber
 \end{equation}

I'll be honest, I did not know what that meant or why it was useful when I was first introduced to it. But it did make sense when I had some visuals so let's do that. This equation is saying the operation $f * g$ is a convolution operation that is defined ($\triangleq$) by repeatedly multiplying a function $f(t)$ with another function $g(t)$ that is delayed by time $\tau$ and therefore making it $g(t-\tau)$ and then adding (or integrating) them together. It might be easier to think of it in discrete space:

\begin{equation}
\label{eq:conv_function_discrete}
  (f * g) [n] \triangleq \sum_{m=-\infty}^{\infty} f(m) g(n - m) \nonumber
\end{equation}

But what does it do and why do we use it in deep learning? Let me use some visual aid - look at this great photo of the NYC skyline:

<figure align="center">
  <img width="700" height="500" src="/assets/img/basics_cnn/nyc_gray.jpg" alt="New York City skyline">
  <!-- <figcaption> Convolution operation is simply glorified multiplication and addition </figcaption> -->
</figure>

To "convolve" over the image means that we treat the image as function $f$ and we use a kernel or a filter function $g$ to go over the image. In other words, this kernel function moves over the image over all the pixels and each time it moves, it multiplies and adds. Usually, kernel sizes are small - 3x3, 5x5, 7x7 and so on. If we use a 3x3 kernel, that means, at a given time $t$, the kernel can cover 9 pixels over the image. It performs element-wise multiplication and then adds them meaning that 3x3 and total of 9 elements gets reduced to a single number. Then the kernel moves $\tau$ steps over to the right, does the same thing. If it reaches the right end of the image, it simply goes down a row and repeats this process until the entire image has been *convolved*. This is show in this gif:

<figure align="center">
  <img width="900" height="400" src="/assets/img/basics_cnn/conv_gif.gif" alt="Convolution operation">
  <figcaption> Convolution operation is simply glorified multiplication and addition </figcaption>
</figure>

So, what does the result look like after the convolution? Well, depending on the kernel, we can do different things. For example, here I chose two kernels:

<figure align="center">
  <img width="250" height="250" src="/assets/img/basics_cnn/kernel1.png" alt="Kernel example">
</figure>


<figure align="center">
  <img width="250" height="250" src="/assets/img/basics_cnn/kernel2.png" alt="Kernel example">
  <figcaption> Examples of kernels/filters </figcaption>
</figure>

When these filters/kernels are convolved over the image, remember that all it does is multiply and add the pixels from the image with pixels or elements of the kernel which results in something called a feature map (result of a convolution):

<figure align="center">
  <img width="900" height="500" src="/assets/img/basics_cnn/filter_result.png" alt="Results of a convolution">
  <figcaption> Result of convolution (called feature map) </figcaption>
</figure>

The image at the top right shows that the kernel has extracted the horizontal features of the image. The horizontal lines on the buildings and skyscrapers are clearly visible and notably, the vertical features are barely there. On the other hand, the image on the bottom right does the exact opposite in extracting the vertical features of the skyscrapers. That is with a simple kernel and the convolution operation. Tweaking the kernel, for example changing the 1s to 2s and -1s to -2s, would make it even more aggressive and pronounced- the possibilities are endless. These kernels shown above are more commonly called Sobel filters which, if you have some background in image processing or computer vision, might be familiar to you. They are used primarily for edge detection and can be used for simple applications like [lane detection for autonomous vehicles](https://github.com/vikasnataraja/Lane-Detection-using-OpenCV-in-Python).

So, now that we know the usefulness of convolutions and kernels, how can we make use of them? Before I get to that question, let me pose another one you may be wondering which is - how do you know what kernels extract what features? More specifically, I just said the possibilities with kernels are endless so how do we know which ones get us to where we want? Well... we don't! We don't actually need to know what numbers in the kernels get us to predicting an image. Confused? Well, let me bring back that quote from LeCun's paper: *"...better pattern recognition systems can be built by relying more on ***automatic learning***, and less on hand-designed heuristics"* with the keyword here being **automatic learning**. The neural network deals with all those possibilities of the kernels because we are in the domain on supervised learning which means given an input image, we already know the ground truth. Using that information, the network changes the kernel values, makes a prediction, checks with the cost function to see if it is being optimized and therefore see how close it got to predicting it correctly and adjusts the values of the kernel accordingly and it does this for hundreds of thousands of backpropagations until the cost function has been optimized (usually minimized). Of course the model needs to see thousands if not tens or hundreds of thousands of these images to learn the appropriate features to be extracted because one image does not generalize to others. Think of it this way - if you have only seen golden retrievers in your entire life then when confronted with a poodle, you wouldn't necessarily know what that is.

But still, how do we make use of them? Well, we know that a simple 3x3 kernel with just 0s and 1s can extract edges so imagine what bigger kernels with other combinations of numbers could achieve. One common trick that CNNs employ is to stack the results of these filters to get a feature map. Often times, 16, 32 or higher number of convolution results are combined to get a deep feature map. Because of that, we can get multiple features (number of kernels used = number of features) at each stage.

So, to summarize, convolutions are simple additions of multiplications and when used over images can extract features. When I say features, I mean things like edges in an image like the one shown above, contours of a curved surface, color differences and so on. The beauty of machine learning and specifically, convolutional neural networks is that you don't have to know what those features need to be to predict something. Think of it this way - if you were told to imagine a dog, you would likely picture a golden retriever and its cute paws, black nose, fur and so on. These then become the features which the CNN ***learns*** to understand over time when you show it pictures of different dogs and tell it that they are dogs. This is termed **feature extraction** which is what a CNN does -  it extracts features with little or no help.

## Layers of a CNN

To truly make use of convolutions and kernels, we use a layered approach in CNNs. If you have ever come across a CNN, chances are you have seen something like this:

<figure align="center">
  <img width="900" height="400" src="/assets/img/basics_cnn/alexnet_arch.png" alt="CNN Layers">
  <figcaption> Architecture of a CNN </figcaption>
</figure>

What are these blocks and what do those terms - stride, max pooling and dense? Well, first of all, this is called the architecture of the convolutional neural network. This a schematic way of representing the layers of a CNN which is a great segue to talk about layers. Layers are simply convolutions of different sizes, different strides at different stages of the network. For example, in this figure, there are 5 **convolutional layers** (the big blocks) after the **input layer** and 3 **dense layers**.

<p><strong>Convolutional layer</strong></p>

Convolutional layers are simply layers that perform convolutions over the given input image or the previous layers. Each convolutional layer will have a fixed kernel size for that layer but not a fixed kernel because remember, the kernel keeps changing as the network learns to extract different features. It is very common to have multiple kernels for each layer because then we can combine those features in the next stage. There is no magic number that is right for the kernel size or the number of kernels - these are dependent on the data and the domain but you will usually see numbers like 16, 32, 64... Therefore, if the input image is say, 100x100 in size, and we use 16 kernels each of size 3x3, we could end up with a feature map that is of the size 96x96x16 (the 96 is because convolution inherently reduces the size at the edges but we can overcome that if we use padding to get a feature map that is the same size as the input). The point of combining multiple conv layers is to extract features and represent the input image at different stages and thereby **capture the spatial context of the input**. Once the input image has been convolved over and we have the features and the feature map, that does not mean we are done extracting them because we keep convolving over that feature map with more kernels and at each stage we can keep increasing the number of kernels from 16 to 32 to 128 and so on. Again, there is no magic formula to determine what could be good and what could be overkill but there are a handful of architectures like AlexNet, DeepLab, VGGNet that work across different domains. The depth indicates the number of features so if we use 16 kernels at the first stage, then we end up extracting 16 features which will then in turn be used to extract more features.

<figure align="center">
  <img width="900" height="500" src="/assets/img/basics_cnn/dogcat.gif" alt="CNN Classification">
  <figcaption> Forward propagation through layers of a CNN for classification </figcaption>
</figure>

The *stride* refers to how the kernel moves over its input. I talked about how a 3x3 kernel covers 9 pixels at a time, convolves over it and moves on. That "moves on" part is decided by the stride length which means if we set a stride=1, the kernel after performing the first convolution will move 1 pixel to the right and repeat. If stride=2, the kernel moves 2 pixels at a time. We can also set multi dimensional strides like stride=(2,2) which means the kernel moves 2 pixels to the right till it reaches the edge and then moves 2 pixels down and starts again. Lower strides are usually more expensive because that means more convolutions but could extract better features whereas higher strides are computationally easier but might not be able to get all the features. Strides of 1 and 2 are the most commonly used ones. The gif of the moving kernel above has a stride of (1,1).

<p><strong>Pooling layer</strong></p>

The *pooling* layer is an intermediate layer that is usually employed after the conv layer to downsample the feature map. But why do we need it and why downsample? Well, think of the example from above - the input image is 100x100 and if it is a color image then that would be account for 30000 pixels and after convolution when it results in 96x96x16, that totals nearly 150,000 parameters and as we go further down, that increases to millions. Unless we have a really good GPU and excellent parallel computing and distribution skills, that is not realistic because it would take forever to process and learn. Using pooling, we can downsample the feature map from 96x96x16 to 48x48x16 meaning now, instead of 150,000 parameters, we only have a fourth of it totaling about 40,000 parameters. We get to 48x48 by simply taking 4 pixels at a time and either averaging them (average pooling) or taking the max (max pooling). Another reason to use pooling and especially max pooling is to force the network to make assumptions about the left out parameters and therefore the domain and learn useful representations of that feature map.

<figure align="center">
  <img width="700" height="500" src="/assets/img/basics_cnn/pooling.jpeg" alt="Results of pooling">
  <figcaption> Result of pooling </figcaption>
</figure>


<p><strong>Dense layer and activation functions</strong></p>

The *dense* layer in a CNN is usually used when the task is classification where the end product is to classify an image to one or more pre-determined categories. For instance, classifying dogs and cats given an image means that the CNN will, at the end, only tell you if the image is of a cat or a dog (it does this probabilistically meaning it will give you the probability of the image being a dog and that of a cat). It is similar with handwritten digit classification as well where the end result is to classify an image with a number to one of 9 digits (0 - 9). So what is the role of the dense layer here? Well, first of all, the dense layers have what are called units which are parameters that denote the size of the output. The dense layer provides a means of learning high-dimensional features quite cheaply and what that means is each unit of the dense layer takes all the input nodes from the previous layer and produces a non-linear output which is why they are often called fully-connected layers or hidden layers. The purpose of a dense layer is to introduce some non-linearity into the network.


<figure align="center">
  <img width="800" height="500" src="/assets/img/basics_cnn/dense_layer.png" alt="Dense layer of a CNN">
  <figcaption> Dense layer (middle layer) in a CNN </figcaption>
</figure>


Mathematically, it is represented by the equation $y = wx + b$ where $w$ is the weight, $x$ is the input and $b$ is the bias. The $w$ which represents the weight is often actually a weight matrix and that combined with the bias $b$ are the trainable nodes which the network must learn. For example, if the number of units is 1024 with an input size of 512, then the network has to learn 1024 * (512+1) = 525,312 parameters. That means half a million parameters are trainable! Using this non-linear activation, the dense layer can learn non-linear representations of the input (To learn why non-linearity is important, [check out this StackExchange discussion](https://stats.stackexchange.com/questions/275358/why-is-increasing-the-non-linearity-of-neural-networks-desired)). There are many options for non-linear functions that can work and these are often called **activation functions**. They are also used the conv layers to bring non-linearity there as well. Some common non-linear activation functions are ReLU (Rectified Linear Unit), sigmoid, and tanh.

ReLU function: $ y = max(0, x) $

<figure align="center">
  <img width="500" height="300" src="/assets/img/basics_cnn/relu.png" alt="ReLU activation function">
  <figcaption> ReLU activation function </figcaption>
</figure>

Sigmoid function: $ y = \sigma (x) = \dfrac{1}{1 + e^{-x}}  $

<figure align="center">
  <img width="400" height="300" src="/assets/img/basics_cnn/sigmoid.png" alt="Sigmoid activation function">
  <figcaption> Sigmoid activation function </figcaption>
</figure>

tanh function: $ y = \dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}} $

<figure align="center">
  <img width="400" height="300" src="/assets/img/basics_cnn/tanh.png" alt="tanh activation function">
  <figcaption> tanh activation function </figcaption>
</figure>

Another interesting feature of the dense layers is that they produce the same output vector for a given input vector. That means that if in one case the input is 9 and the output produced by $y = wx + b$ is 16, then for every input case of 9, the output will always be 16 meaning dense layers cannot sense changes different answers for the same input. In cases where you don't need to catch repetition or have different sets of answers for the same input, this is not an issue.

<p><strong>Output layer</strong></p>

The *output layer* is typically a fully-connected or dense layer at the very end that is used in classification tasks to represent class scores. Going back to that example of classifying cats and dogs, this is the layer that would contain and produce the probabilistic score of say 80% that the image is that of a dog and 20% that the image is a cat. But how does the dense layer accomplish that since ReLU activation or tanh activation can yield values >1? Well, one possibility is to use the sigmoid function which naturally maps the output of the previous layer to a probability score between 0 and 1. Another popular activation function is the softmax function which is given by:

\begin{equation}
\label{eq:softmax_function}
  y = \sigma(x_{i}) = \dfrac{e^{x_{i}}}{\sum_{j} e^{x_{j}}} \nonumber
\end{equation}  


$softmax$ scales the output of the previous layer to probability scores that total 1. You will notice that the sigmoid and softmax look very similar and that is because they are. The difference is that sigmoid treats its outputs or classes independently meaning the probability scores of 2 output classes can be 0.5 and 0.3 or 0.7 and 0.1 where the sum of the probabilities don't have to add up to 1. In the case of softmax, the sum of probabilties across the classes will yield 1. This makes it easy when you have to classify an image into one of many classes such as the handwritten digit classification with 9 classes where the image can only be one number and one number only and so only that class probability will be maximized (this is also why it is called maximum entropy classifier). In cases where you might have an image that could belong to one or more classes at the same time, the sigmoid function can be used. For instance, facial expression classification - a person can at the same time have raised eyebrows and be smiling but using the softmax function would only capture one of those if at all. Using sigmoid, we can overcome that problem. Check out [my work on handwritten digit classification using TensorFlow](https://github.com/vikasnataraja/Handwritten-Digit-Classification-using-TensorFlow) with a very simple architecture of 2 convolutional layers and 1 hidden layer!


**tl;dr - use softmax when the input image can only belong to one class, use softmax when the input image has multiple classes or can be classified into multiple categories.**


<figure align="center">
  <img width="600" height="500" src="/assets/img/basics_cnn/digits.png" alt="Handwritten digits">
  <figcaption> Handwritten digit dataset from MNIST </figcaption>
</figure>

All these layers - input layer, convolutional layer, pooling layer, dense or hidden layer and the output layer - together with the activation functions combine to form a Convolutional Neural Network. The depth is left to you but of course the deeper you go, the more computationally expensive it get and the more time it takes. It also might not be necessary for the problem at hand to go too deep. For example, problems like classifying a dog and a cat can be done with just 3-4 convolutional layers and a couple of dense layers. These days, architectures are incredibly deep going up to 250 layers or more. There are very efficient ways to stack more layers and yet retain computational speed (see ResNet, DeepLab) but I'll talk about that in another blog. For now, keep exploring the fascinating world of CNNs - it's not very complex and all the concepts are rooted in math :)


Cover Image credit: Gertrūda Valasevičiūtė via UnSplash

# fitted-learning
This is a small Keras implementation of Fitted Learning: Models with Awareness of their Limits.
It shows how to train a simple network that will over-generalize less than a more standard network.

The dataset consists of point coordinates (x,y) from circles of different radius,as show in the image below
![dataset](http://i.imgur.com/kMsvO6m.png)

The parameter DOO can be changed to tune the degree of generalization of the network, with smaller DOO meaning more generalization. 
A DOO of 1 is equivalent to a standard NN with a softmax layer and crossentropy loss. The following images show how the space is classified by the NN for different values of DOO.
The dataset is shown in white rather than the original colors for visibility.

Class 1 is blue, class 2 is green, class 3 is red. The black space shows regions where the probability is low for all classes.

DOO=1 (standard NN)

![1](http://i.imgur.com/yI17Euk.png)


DOO=2

![2](http://i.imgur.com/RRYHup6.png)


DOO=6

![6](http://i.imgur.com/OmH56Vm.png)


DOO=24

![24](http://i.imgur.com/kroBjfe.png)


DOO=48

![48](http://i.imgur.com/Uasd2wP.png)

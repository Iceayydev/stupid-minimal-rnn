# stupid rnn

when i was starting to write this i was originally gonna go for a perceptron, but the youtube algoritm blessed me with a long ass video on rnns and i decided to try it for myself. and now we're here.

we can look at the code in this structure:

* **data preprocessing**
* **model definition**
* **model training**
* **model prediction**

## data preprocessing
first, we convert everything to lowercase, splitting them into words, and creating a vocabulary of unique words. we also create a dictionary to map its index in our vocabulary.

## rnn definition
our input size is equal to our vocab size
our hidden size is 10
output size is equal to our vocab size
we use tanh activation for our beloved hidden layer
silly softmax output for prob distribution over the next word

our weights are initialized randomly
- wxh - input to hidden
- whh - hidden to hidden
- why - hidden to output (lmao it spelled out a word hahahahahaha)

## training

we're training on 100 epochs here, and i used stochastic gradient descent and categorical cross-entropy loss.
### stochastic gradient (just so you remember what it is)
- so we can minimize the loss function
- avoid getting the true gradient using the whole dataset, so we estimate with a bunch of batch samples
- makes it a bit faster
- updates weights in the estimated gradient direction so we can minimize loss

anyways,
the forward pass here calculates the hidden state and output prediction
the backward pass computes the weight gradients using backprop through TIME!!!!!!
we update (the weights) with a 0.1 learning rate.

after we're done training, we can finally use it to predict the next word, from the input sequence!!

our code here uses 3 `the car goes fast` and one `the car goes slow`, i think the objective here is obvious but if you got this far you're probably foaming from all the college cources you've been taking computers pretending to talk so:

the hope is the model selects fast as our output, since it appears more than slow. if we were to reverse everything, and make slow appear more than fast, we would get `slow` more. if we made them equal, we get fucking anything but `fast` or `slow`.

anyways im tired from all the shape not aligned issues, so have fun? even though this has been done like 500 times, and i guess mine is more "for one purpose" or whatever.

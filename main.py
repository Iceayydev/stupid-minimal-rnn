import numpy as np

# training data!!!!!!!!
training_data = ["the car goes fast", "the car goes fast", "the car goes fast", "the car goes slow"]

# our preprocessing
corpus = ' '.join(training_data).lower().split()
vocab = list(set(corpus))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
num_words = len(vocab)

# create input and target seq
seq_length = 3
X = []
y = []
for i in range(len(corpus) - seq_length):
    input_seq = corpus[i:i + seq_length]
    target_word = corpus[i + seq_length]
    X.append([word_to_idx[word] for word in input_seq])
    y.append(word_to_idx[target_word])

# convert input, target sequences to np arrays
X = np.array(X)
y = np.array(y)

# models params
hidden_size = 10
learning_rate = 0.1
epochs = 100
print_specifics = False  # turn this on if you wanna see the weights, and hidden state... for some reason.


def print_specificsdef():
    print(f"[hidden state]----------------------------------------------")
    print(h)
    print(f"[hidden size]-----------------------------------------------")
    print(hidden_size)
    print(f"[input to hidden]-------------------------------------------")
    print(Wxh)
    print(f"[hidden to hidden]------------------------------------------")
    print(Whh)
    print(f"[hidden to output]------------------------------------------")
    print(Why)


# init weights
Wxh = np.random.randn(hidden_size, num_words) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(num_words, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((num_words, 1))

# our beloved hidden state
h = np.zeros((hidden_size, 1))

# training loop
for epoch in range(epochs):
    # forward pass
    loss = 0
    for t in range(len(X)):
        x = np.zeros((num_words, 1))
        x[X[t]] = 1

        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y_pred = np.dot(Why, h) + by
        probs = np.exp(y_pred) / np.sum(np.exp(y_pred))

        loss += -np.log(probs[y[t], 0])

    # our backward pass
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)
    dh_next = np.zeros_like(h)

    for t in reversed(range(len(X))):
        x = np.zeros((num_words, 1))
        x[X[t]] = 1

        dy = np.copy(probs)
        dy[y[t]] -= 1

        dWhy += np.dot(dy, h.T)
        dby += dy

        dh = np.dot(Why.T, dy) + dh_next
        dh_raw = (1 - h * h) * dh

        dbh += dh_raw
        dWxh += np.dot(dh_raw, x.T)
        dWhh += np.dot(dh_raw, h.T)

        dh_next = np.dot(Whh.T, dh_raw)

    # we update weights
    Wxh -= learning_rate * dWxh
    Whh -= learning_rate * dWhh
    Why -= learning_rate * dWhy
    bh -= learning_rate * dbh
    by -= learning_rate * dby

    # printing epoch and loss
    if epoch % 10 == 0:
        print(f"epoch: {epoch}, loss: {loss}")
    if print_specifics == True:
        print_specificsdef()

# get next word
input_seq = "the car goes"
input_seq = input_seq.lower().split()
x = np.zeros((num_words, 1))
x[[word_to_idx[word] for word in input_seq]] = 1

h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
y_pred = np.dot(Why, h) + by
probs = np.exp(y_pred) / np.sum(np.exp(y_pred))

next_word_idx = np.argmax(probs)
next_word = idx_to_word[next_word_idx]

print(f"the car goes {next_word}")

if print_specifics ==  True:
    print_specificsdef()


import numpy as np
import csv
import os

model_count = 100
loss_path = "data/loss"
word_path = "data/word"

if os.path.exists(loss_path):
    for item in os.listdir(loss_path):
        os.remove(f"{loss_path}/{item}")
else:
    os.makedirs(loss_path)

if os.path.exists(word_path):
    for item in os.listdir(word_path):
        os.remove(f"{word_path}/{item}")
else:
    os.makedirs(word_path)

for cycle in range(model_count):

    with open(f"{loss_path}/result{cycle}.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["epoch","loss"]
        writer.writerow(header)

    # training data!!!!!!!!
    training1 = ["the car goes fast", "that duck moves fast", "those turtles are fast", "them tires move slow"]
    training_data = training1

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


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    # init weights for lstm
    Wf = np.random.randn(hidden_size, num_words) * 0.01
    Uf = np.random.randn(hidden_size, hidden_size) * 0.01
    bf = np.zeros((hidden_size, 1))

    Wi = np.random.randn(hidden_size, num_words) * 0.01
    Ui = np.random.randn(hidden_size, hidden_size) * 0.01
    bi = np.zeros((hidden_size, 1))

    Wo = np.random.randn(hidden_size, num_words) * 0.01
    Uo = np.random.randn(hidden_size, hidden_size) * 0.01
    bo = np.zeros((hidden_size, 1))

    Wc = np.random.randn(hidden_size, num_words) * 0.01
    Uc = np.random.randn(hidden_size, hidden_size) * 0.01
    bc = np.zeros((hidden_size, 1))

    Why = np.random.randn(num_words, hidden_size) * 0.01
    by = np.zeros((num_words, 1))

    # our beloved hidden state and cell state
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))

    # training loop
    for epoch in range(epochs):
        # forward pass
        loss = 0
        for t in range(len(X)):
            x = np.zeros((num_words, 1))
            x[X[t]] = 1

            ft = sigmoid(np.dot(Wf, x) + np.dot(Uf, h) + bf)
            it = sigmoid(np.dot(Wi, x) + np.dot(Ui, h) + bi)
            ot = sigmoid(np.dot(Wo, x) + np.dot(Uo, h) + bo)
            ct = ft * c + it * np.tanh(np.dot(Wc, x) + np.dot(Uc, h) + bc)
            h = ot * np.tanh(ct)

            y_pred = np.dot(Why, h) + by
            probs = np.exp(y_pred) / np.sum(np.exp(y_pred))

            loss += -np.log(probs[y[t], 0])

        # our backward pass
        dWf = np.zeros_like(Wf)
        dUf = np.zeros_like(Uf)
        dbf = np.zeros_like(bf)

        dWi = np.zeros_like(Wi)
        dUi = np.zeros_like(Ui)
        dbi = np.zeros_like(bi)

        dWo = np.zeros_like(Wo)
        dUo = np.zeros_like(Uo)
        dbo = np.zeros_like(bo)

        dWc = np.zeros_like(Wc)
        dUc = np.zeros_like(Uc)
        dbc = np.zeros_like(bc)

        dWhy = np.zeros_like(Why)
        dby = np.zeros_like(by)

        dh_next = np.zeros_like(h)
        dc_next = np.zeros_like(c)

        for t in reversed(range(len(X))):
            x = np.zeros((num_words, 1))
            x[X[t]] = 1

            ft = sigmoid(np.dot(Wf, x) + np.dot(Uf, h) + bf)
            it = sigmoid(np.dot(Wi, x) + np.dot(Ui, h) + bi)
            ot = sigmoid(np.dot(Wo, x) + np.dot(Uo, h) + bo)
            ct = ft * c + it * np.tanh(np.dot(Wc, x) + np.dot(Uc, h) + bc)
            ht = ot * np.tanh(ct)

            dy = np.copy(probs)
            dy[y[t]] -= 1

            dWhy += np.dot(dy, ht.T)
            dby += dy

            dh = np.dot(Why.T, dy) + dh_next

            dot = dh * np.tanh(ct)
            dot_input = dot * ot * (1 - ot)
            dWo += np.dot(dot_input, x.T)
            dUo += np.dot(dot_input, h.T)
            dbo += dot_input

            dc = (dh * ot) * (1 - np.tanh(ct) ** 2) + dc_next
            dft = dc * c * ft * (1 - ft)
            dit = dc * np.tanh(np.dot(Wc, x) + np.dot(Uc, h) + bc) * it * (1 - it)
            dct = dc * it * (1 - np.tanh(np.dot(Wc, x) + np.dot(Uc, h) + bc) ** 2)

            dWf += np.dot(dft, x.T)
            dUf += np.dot(dft, h.T)
            dbf += dft

            dWi += np.dot(dit, x.T)
            dUi += np.dot(dit, h.T)
            dbi += dit

            dWc += np.dot(dct, x.T)
            dUc += np.dot(dct, h.T)
            dbc += dct

            dh_next = np.dot(Uf.T, dft) + np.dot(Ui.T, dit) + np.dot(Uo.T, dot_input)
            dc_next = dc * ft

        # update lstm param
        Wf -= learning_rate * dWf
        Uf -= learning_rate * dUf
        bf -= learning_rate * dbf

        Wi -= learning_rate * dWi
        Ui -= learning_rate * dUi
        bi -= learning_rate * dbi

        Wo -= learning_rate * dWo
        Uo -= learning_rate * dUo
        bo -= learning_rate * dbo

        Wc -= learning_rate * dWc
        Uc -= learning_rate * dUc
        bc -= learning_rate * dbc

        Why -= learning_rate * dWhy
        by -= learning_rate * dby

        with open(f"{loss_path}/result{cycle}.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            row = [epoch,loss]
            writer.writerow(row)

        # printing epoch and loss
        if epoch % 10 == 0:
            print(f"epoch: {epoch}, loss: {loss}")

    
    # get next word
    input_seq = "the car goes"
    inp_txt = input_seq
    input_seq = input_seq.lower().split()
    x = np.zeros((num_words, 1))
    x[[word_to_idx[word] for word in input_seq]] = 1

    ft = sigmoid(np.dot(Wf, x) + np.dot(Uf, h) + bf)
    it = sigmoid(np.dot(Wi, x) + np.dot(Ui, h) + bi)
    ot = sigmoid(np.dot(Wo, x) + np.dot(Uo, h) + bo)
    ct = ft * c + it * np.tanh(np.dot(Wc, x) + np.dot(Uc, h) + bc)
    h = ot * np.tanh(ct)

    y_pred = np.dot(Why, h) + by
    probs = np.exp(y_pred) / np.sum(np.exp(y_pred))

    next_word_idx = np.argmax(probs)
    next_word = idx_to_word[next_word_idx]

    print(f"--[ {str(inp_txt)} > {next_word} < ]--")

    with open(f"{word_path}/result{cycle}.txt", "w") as txtfile:
        txtfile.write(next_word)

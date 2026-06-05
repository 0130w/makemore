import logging
import random
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class Linear:
    def __init__(self, in_features, out_features, useBias=False):
        self.weight = torch.randn(in_features, out_features) / in_features**0.5
        self.bias = torch.zeros(out_features) if useBias else None
        self.training = False

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias else [])


class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            x_mean = x.mean(0, keepdim=True)
            x_var = x.var(0, keepdim=True)
        else:
            x_mean = self.running_mean
            x_var = self.running_var
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        self.out = x_hat * self.gamma + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * x_mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * x_var
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __init__(self):
        self.training = False

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
        self.training = False

    def __call__(self, x):
        return self.weight[x]

    def parameters(self):
        return [self.weight]


class Flatten:
    def __init__(self) -> None:
        self.training = False

    def __call__(self, x):
        return x.view(x.shape[0], -1)  # x.shape[0] is batch_size

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.weight *= 5 / 3

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def main():
    words = open("names.txt", "r").read().lower().splitlines()
    # create training set for n-gram model
    chars = sorted(list(set("".join(words))))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    embed_size = 16
    hidden_size_1 = 300
    block_size = 8
    batch_size = 32
    vocab_size = len(stoi)

    def build_dataset(words):
        X, Y = [], []
        for w in words:
            context = [0] * block_size
            w += "."
            for ch in w:
                idx = stoi[ch]
                X.append(context)
                Y.append(idx)
                context = context[1:] + [idx]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    X_tr, Y_tr = build_dataset(words[:n1])
    X_dev, Y_dev = build_dataset(words[n1:n2])
    X_te, Y_te = build_dataset(words[n2:])
    model = Sequential(
        [
            Embedding(vocab_size, embed_size),
            Flatten(),
            Linear(embed_size * block_size, hidden_size_1),
            BatchNorm1D(hidden_size_1),
            Tanh(),
            Linear(hidden_size_1, hidden_size_1),
            BatchNorm1D(hidden_size_1),
            Tanh(),
            Linear(hidden_size_1, hidden_size_1),
            BatchNorm1D(hidden_size_1),
            Tanh(),
            Linear(hidden_size_1, vocab_size),
            BatchNorm1D(vocab_size),
        ]
    )
    parameters = model.parameters()
    for p in parameters:
        p.requires_grad = True
    iter_num = 50000

    i_samples, loss_samples = [], []

    # train
    for i in range(iter_num):
        idx = torch.randint(0, X_tr.shape[0], (batch_size,))
        X_batch, Y_batch = X_tr[idx], Y_tr[idx]
        x = model(X_batch)
        loss = F.cross_entropy(x, Y_batch)
        for p in parameters:
            p.grad = None
        loss.backward()
        lr = 0.1 if i < 1000 else 0.01
        for p in parameters:
            assert p.grad is not None, f"{p} grad is None"
            p.data += -lr * p.grad  # need learning rate decay
        i_samples.append(i)
        loss_samples.append(loss.item())

    loss_tensor = torch.tensor(loss_samples)
    smooth_loss = loss_tensor.view(-1, 200).mean(1)
    smooth_x = torch.arange(len(smooth_loss)) * 200
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.plot(
        i_samples, loss_samples, label="Original Loss", color="lightgray", alpha=0.6
    )
    plt.plot(
        smooth_x.numpy(),
        smooth_loss.numpy(),
        label="Smoothed Loss",
        color="blue",
        linewidth=2,
    )
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    for layer in model.layers:
        layer.training = False

    # evaluate loss on dev split
    x = model(X_dev)
    loss = F.cross_entropy(x, Y_dev)
    print(f"loss = {loss}")


if __name__ == "__main__":
    main()

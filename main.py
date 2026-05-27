import sys
import torch
import random
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


class Linear:
    def __init__(self, in_features, out_features, useBias=False):
        self.weight = torch.randn(in_features, out_features) / in_features**0.5
        self.bias = torch.zeros(out_features) if useBias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias else [])


class BatchNorm1D:
    def __init__(self):
        pass

    def __call__(self):
        pass

    def parameters(self):
        pass


def main():
    words = open("names.txt", "r").read().lower().splitlines()
    # create training set for bi-gram model
    chars = sorted(list(set("".join(words))))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    embed_size = 16
    hidden_size_1 = 300
    block_size = 8
    batch_size = 32

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
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn([27, embed_size], generator=g)
    gain_W_1 = 5 / 3
    W_1 = (
        torch.randn((block_size * embed_size, hidden_size_1), generator=g)
        * gain_W_1
        / (block_size * embed_size) ** 0.5
    )
    b_1 = torch.zeros(hidden_size_1)
    W_2 = torch.randn(hidden_size_1, 27, generator=g) * 0.01
    b_2 = torch.zeros(27)

    b_gain = torch.ones(1, hidden_size_1)
    b_mean_running = torch.ones(1, hidden_size_1)
    b_std_running = torch.zeros(1, hidden_size_1)
    b_bias = torch.zeros(1, hidden_size_1)

    parameters = [W_1, b_1, W_2, b_2, C]
    for p in parameters:
        p.requires_grad = True
    iter_num = 50000

    i_samples, loss_samples = [], []

    # train
    for i in range(iter_num):
        # create minibatch
        # idx shape: [batch_size]
        idx = torch.randint(0, X_tr.shape[0], (batch_size,))
        # one-hot encoding
        # X_tr[idx] shape: [batch_size, block_size]
        # C[X_tr[idx]] shape: [batch_size, block_size, embed_size]
        # 其实用什么做索引，尺寸就会变成索引的那个矩阵的尺寸放前面，被索引的尺寸放后面
        x_emb = C[X_tr[idx]]  # high dimension tensor index, shape: X.shape + C.shape.1
        # out = torch.cat(torch.unbind(x_emb, 1), dim=1)    # ineffient oper torch.cat
        # IDEA: want this activation be Gaussion
        h_preact = x_emb.view(-1, block_size * embed_size) @ W_1 + b_1
        h_mean = h_preact.mean(0, keepdim=True)
        h_std = h_preact.std(0, keepdim=True)
        h_preact = b_gain * (h_preact - h_mean) / h_std + b_bias
        with torch.no_grad():  # used for inferring
            b_mean_running = 0.999 * b_mean_running + 0.001 * h_mean
            b_std_running = 0.999 * b_std_running + 0.001 * h_std
        out = torch.tanh(h_preact)
        logits = out @ W_2 + b_2  # batch_size, 27
        loss = F.cross_entropy(
            logits, Y_tr[idx]
        )  # fused kernel: efficent and numerical stability
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

    # evaluate loss on dev split
    x_emb = C[X_dev]
    out = torch.tanh(x_emb.view(-1, block_size * embed_size) @ W_1 + b_1)
    logits = out @ W_2 + b_2
    loss = F.cross_entropy(logits, Y_dev)
    print(f"loss = {loss}")


if __name__ == "__main__":
    main()

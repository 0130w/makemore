import torch
import torch.nn.functional as F

def base():
    # bi-gram use previous one char to predict next one
    words = open("names.txt", "r").read().lower().splitlines()
    chars = sorted(list(set(''.join(words))))
    stoi = { ch : idx + 1 for idx, ch in enumerate(chars) }   # character -> idx
    stoi['.'] = 0   # just tricky way to avoid all zero row/col
    cnt = len(stoi)
    N = torch.zeros((cnt, cnt), dtype=torch.int32)
    for w in words:
        w = '.' + w + '.'
        for ch1, ch2 in zip(w, w[1:]):
            idx1 = stoi[ch1]
            idx2 = stoi[ch2]
            N[idx1, idx2] += 1
    # model smoothing
    P = (N + 1).float()   # -> model param
    P /= P.sum(1, keepdim=True)  # broadcast rule [27 * 27] / [27 * 1] -> here broadcast is necessary
    seed = 2147483647
    g = torch.Generator().manual_seed(seed)
    idx = 0 # starts from '.'
    while True:
        p = P[idx]
        num_samples = 1
        # replacement = True -> 有放回抽样
        sample_idx = int(torch.multinomial(p, num_samples, replacement=True, generator=g).item())
        if sample_idx == 0:
            break
        idx = sample_idx

    # use likelihood(log likelihood) to evaluate model
    log_likelihood = 0.0
    n = 0
    for w in words:
        for ch1, ch2 in zip(w, w[1:]):
            idx1 = stoi[ch1]
            idx2 = stoi[ch2]
            prob = P[idx1, idx2]
            log_likelihood += torch.log(prob)
            n += 1

    # minimize the loss
    loss = -log_likelihood / n
    print(f"loss = {loss}")

def main():
    words = open("names.txt", 'r').read().lower().splitlines()
    # create training set for bi-gram model
    chars = sorted(list(set(''.join(words))))
    stoi = { ch : i + 1 for i, ch in enumerate(chars) }
    stoi['.'] = 0
    itos = { i : ch for ch, i in stoi.items()}

    block_size = 4  # use 3 characters to predict next one
    embed_size = 2
    hidden_size_1 = 100
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        w += '.'
        for ch in w:
            idx = stoi[ch]
            X.append(context)
            Y.append(idx)
            context = context[1:] + [idx]
    g = torch.Generator().manual_seed(2147483647)
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    C = torch.randn([27, embed_size], generator=g)
    W_1 = torch.randn((block_size * embed_size, hidden_size_1), generator=g)
    b_1 = torch.randn(hidden_size_1, generator=g)
    W_2 = torch.randn(hidden_size_1, 27, generator=g)
    b_2 = torch.randn(27, generator=g)
    parameters = [W_1, b_1, W_2, b_2, C]
    for p in parameters:
        p.requires_grad = True
    alpha = 1
    for _ in range(100):
        # create minibatch
        idx = torch.randint(0, X.shape[0], (32,))
        # one-hot encoding
        x_emb = C[X[idx]]    # high dimension tensor index, shape: X.shape + C.shape.1
        # out = torch.cat(torch.unbind(x_emb, 1), dim=1)    # ineffient oper torch.cat
        out = torch.tanh(x_emb.view(-1, block_size * embed_size) @ W_1 + b_1)
        logits = out @ W_2 + b_2 # batch_size, 27
        loss = F.cross_entropy(logits, Y[idx])   # fused kernel: efficent and numerical stability
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            assert p.grad != None, f"{p} grad is None"
            p.data += -alpha * p.grad
        print(f"loss = {loss}")


if __name__ == "__main__":
    main()

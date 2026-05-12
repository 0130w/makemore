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
    xs, ys = [], []
    for w in words:
        for ch1, ch2 in zip(w, w[1:]):
            idx1 = stoi[ch1]
            idx2 = stoi[ch2]
            xs.append(idx1) # input : first char index
            ys.append(idx2) # label : second char index
    
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.nelement()
    print(f"xs = {xs}\n ys = {ys.shape}")

    # one-hot encoding
    x_enc = F.one_hot(xs, num_classes=len(stoi)).float()    # shape : len(stoi)

    alpha = 50
    W = torch.randn([27, 27], requires_grad=True)
    for _ in range(100):
        # forward pass
        logits = x_enc @ W # -> log count (why interpret as log count)
        cnt = logits.exp()
        probs = cnt / cnt.sum(1, keepdim=True)
        index = torch.arange(num)
        loss = -probs[index, ys].log().mean() + 0.01 * (W**2).mean() # regulization(similar as smoothing)
        # clear grad
        W.grad = None
        loss.backward()
        assert W.grad != None, "W gradient is None"
        W.data += -alpha * W.grad
        print(f"loss = {loss}")

    # sample
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(5):
        out = ""
        idx = 0
        while True:
            x_enc = F.one_hot(torch.tensor([idx]), num_classes=27).float()
            logits = x_enc @ W
            cnt = logits.exp()
            probs = cnt / cnt.sum(1, keepdim=True)
            sample_idx = torch.multinomial(probs, 1, replacement=True, generator=g).item()
            if sample_idx == 0:
                break
            out += itos[sample_idx] # type: ignore
        print(f"out = {out}")


if __name__ == "__main__":
    main()

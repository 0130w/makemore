import torch
import seaborn as sns
import matplotlib.pyplot as plt


def draw(matrix, xlabels, ylabels):
    plt.figure(figsize=(24, 24))
    sns.heatmap(
        matrix.numpy(),
        annot=True,
        fmt=".1g",
        cmap="Reds",
        xticklabels=xlabels,
        yticklabels=ylabels,
    )
    plt.title("BiGram Probabilities", fontsize=20)
    plt.xlabel("Second Character", fontsize=20)
    plt.ylabel("First Character", fontsize=20)
    plt.savefig("P_heatmap.png")
    plt.show()


def sample(itos, P):
    idx = 0
    ctx = ""
    seed = 123
    num_samples = 1
    g = torch.Generator().manual_seed(seed)
    while True:
        p = P[idx]
        sample_idx = int(
            torch.multinomial(p, num_samples, replacement=True, generator=g)
        )
        if sample_idx == 0:
            break
        else:
            ctx += itos[sample_idx]
            idx = sample_idx


def main():
    words = open("names.txt", "r").read().lower().splitlines()
    chars = sorted(list(set("".join(words))))
    stoi = {ch: idx + 1 for idx, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {idx: ch for ch, idx in stoi.items()}
    num = len(stoi)
    N = torch.zeros([num, num], dtype=torch.int32)
    for word in words:
        word = "." + word + "."
        for ch1, ch2 in zip(word, word[1:]):
            idx1 = stoi[ch1]
            idx2 = stoi[ch2]
            N[idx1, idx2] += 1
    P = N.float()
    P = P / P.sum(1, keepdim=True)
    log_P = torch.log(P)
    loss = 0
    for word in words:
        word = "." + word + "."
        for ch1, ch2 in zip(word, word[1:]):
            idx1 = stoi[ch1]
            idx2 = stoi[ch2]
            loss += log_P[idx1, idx2]
    loss = loss / len(words)
    print(loss)


if __name__ == "__main__":
    main()

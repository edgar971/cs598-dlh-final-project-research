import torch


class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        torch.nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

        self.sharedWeights = torch.nn.Linear(emb_dim, vocab_size)
        self.sharedWeights.weight.data = self.embedding.weight.data

    def get_embedding(self, multi_hot):
        return self.embedding(multi_hot)

    def get_decode2vocab(self, inputs):
        return self.sharedWeights(inputs)

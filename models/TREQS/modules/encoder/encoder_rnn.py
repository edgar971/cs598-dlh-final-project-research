import torch
from torch.autograd import Variable


class EncoderRNN(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        hidden_size,
        nLayers,
        device=torch.device("cpu"),
    ):
        """
        RNN encoder
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.nLayers = nLayers
        self.device = device

        self.encoder = torch.nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=nLayers,
            batch_first=True,
            bidirectional=True,
        ).to(device)

    def forward(self, input_):
        """
        returns encoding
        """
        n_dk = 2
        batch_size = input_.shape[0]

        h0_encoder = Variable(
            torch.zeros(n_dk * self.nLayers, batch_size, self.hidden_size)
        ).to(self.device)

        c0_encoder = Variable(
            torch.zeros(n_dk * self.nLayers, batch_size, self.hidden_size)
        ).to(self.device)

        hy_encoder, (ht_encoder, ct_encoder) = self.encoder(
            input_, (h0_encoder, c0_encoder)
        )

        return hy_encoder, (ht_encoder, ct_encoder)

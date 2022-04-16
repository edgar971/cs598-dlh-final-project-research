import torch
from torch.autograd import Variable

from TREQS.modules.encoder.encoder_rnn import EncoderRNN
from TREQS.modules.encoder_to_decoder.encoder_to_decoder import EncoderToDecoder


class MainEncoder(torch.nn.Module):
    def __init__(self, args, embedding):
        super(MainEncoder, self).__init__()

        self.args = args
        self.embedding = embedding

        self.encoder = EncoderRNN(
            self.args["emb_dim"],
            self.args["src_hidden_dim"],
            self.args["nLayers"],
            device=self.args["device"],
        ).to(self.args["device"])

        self.encoder_to_decoder = EncoderToDecoder(
            src_hidden_size=self.args["src_hidden_dim"],
            trg_hidden_size=self.args["trg_hidden_dim"],
        ).to(self.args["device"])

    def forward(self, data):
        batch_size = data.shape[0]

        x_embed = self.embedding.get_embedding(data)
        x_enc, x_hidden = self.encoder(x_embed)
        trg_hidden0 = self.encoder_to_decoder(x_hidden)

        h_attn = Variable(torch.zeros(batch_size, self.args["trg_hidden_dim"])).to(
            self.args["device"]
        )

        past_attn = Variable(
            torch.ones(batch_size, self.args["src_seq_len"])
            / float(self.args["src_seq_len"])
        ).to(self.args["device"])

        past_dech = Variable(torch.zeros(1, 1)).to(self.args["device"])

        return x_embed, x_enc, trg_hidden0, h_attn, past_attn, past_dech

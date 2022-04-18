import torch

from TREQS.modules.attention.nats_attention_encoder import AttentionEncoder
from TREQS.modules.attention.nats_attention_decoder import AttentionDecoder


class MainDecoder(torch.nn.Module):
    def __init__(self, args):
        super(MainDecoder, self).__init__()

        self.args = args

        self.decoderRNN = torch.nn.LSTMCell(
            self.args["emb_dim"] + self.args["trg_hidden_dim"],
            self.args["trg_hidden_dim"],
        ).to(self.args["device"])

        self.attnEncoder = AttentionEncoder(
            self.args["src_hidden_dim"],
            self.args["trg_hidden_dim"],
            attn_method="luong_general",
            repetition="temporal",
        ).to(self.args["device"])

        self.attnDecoder = AttentionDecoder(
            self.args["trg_hidden_dim"], attn_method="luong_general"
        ).to(self.args["device"])

        self.wrapDecoder = torch.nn.Linear(
            self.args["src_hidden_dim"] * 2 + self.args["trg_hidden_dim"] * 2,
            self.args["trg_hidden_dim"],
            bias=True,
        ).to(self.args["device"])

        self.genPrb = torch.nn.Linear(
            self.args["emb_dim"]
            + self.args["src_hidden_dim"] * 2
            + self.args["trg_hidden_dim"],
            1,
        ).to(self.args["device"])

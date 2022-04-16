import torch


class EncoderToDecoder(torch.nn.Module):
    def __init__(self, src_hidden_size, trg_hidden_size):
        """
        encoder rnn 2 decoder rnn.
        """
        super(EncoderToDecoder, self).__init__()

        self.encoder_to_decoder = torch.nn.Linear(2 * src_hidden_size, trg_hidden_size)
        self.encoder_to_decoder_c = torch.nn.Linear(
            2 * src_hidden_size, trg_hidden_size
        )

    def forward(self, hidden_encoder):
        (src_h_t, src_c_t) = hidden_encoder
        h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)

        decoder_h0 = torch.tanh(self.encoder_to_decoder(h_t))
        decoder_c0 = torch.tanh(self.encoder_to_decoder_c(c_t))

        return (decoder_h0, decoder_c0)

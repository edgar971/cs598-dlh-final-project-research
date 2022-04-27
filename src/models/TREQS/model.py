import os

import torch
from torch.autograd import Variable

from TREQS.data.utils import construct_vocab
from TREQS.data.seq2sql.process_batch_cqa_v1 import process_batch

from TREQS.modules.embedding.embedding import Embedding
from TREQS.modules.attention.nats_attention_encoder import AttentionEncoder
from TREQS.modules.attention.nats_attention_decoder import AttentionDecoder
from TREQS.base import BaseModel
from TREQS.modules.encoder.main_encoder import MainEncoder
from src.models.TREQS.modules.decoder.main_decoder import MainDecoder


class TREQS(BaseModel):
    def __init__(self, args):
        self.pipe_data = {}  # for pipe line
        self.beam_data = []  # for beam search
        super().__init__(args=args)

    def build_vocabulary(self):
        """
        build vocabulary
        """
        vocab2id, id2vocab = construct_vocab(
            file_=os.path.join(self.args["data_dir"], self.args["file_vocab"]),
            max_size=self.args["max_vocab_size"],
            mincount=self.args["word_minfreq"],
        )

        vocab_size = len(vocab2id)
        self.batch_data["vocab2id"] = vocab2id
        self.batch_data["id2vocab"] = id2vocab
        self.batch_data["vocab_size"] = vocab_size
        print("The vocabulary size: {}".format(vocab_size))

    def build_optimizer(self, params):
        """
        Build model optimizer
        """
        optimizer = torch.optim.Adam(params, lr=self.args["learning_rate"])

        return optimizer

    def init_train_model_params(self):
        """
        Initialize Train Model Parameters.
        For testing and visulization.
        """
        for model_name in self.train_models:
            fl_ = os.path.join(
                self.args["train_model_dir"],
                model_name + "_" + str(self.args["best_model"]) + ".model",
            )
            self.train_models[model_name].load_state_dict(
                torch.load(fl_, map_location=lambda storage, loc: storage)
            )

    def beam_search(self):
        """
        Light Weight Beam Search Algorithm.
        @author Ping Wang and Tian Shi
        Please contact ping@vt.edu or tshi@vt.edu
        Original repo from: https://github.com/wangpinggl/TREQS
        """
        self.encode()
        lastwd = Variable(torch.LongTensor([self.batch_data["vocab2id"]["select"]])).to(
            self.args["device"]
        )
        self.beam_data = [[[lastwd], 0.0, self.pipe_data["decoderB"]]]

        for k in range(self.args["trg_seq_len"]):
            beam_tmp = []
            for j in range(len(self.beam_data)):
                lastwd = self.beam_data[j][0][-1]
                if lastwd == self.batch_data["vocab2id"]["<stop>"]:
                    beam_tmp.append(self.beam_data[j])
                    continue
                self.pipe_data["decoderA"] = self.beam_data[j][2]
                if lastwd >= len(self.batch_data["vocab2id"]):
                    lastwd = Variable(
                        torch.LongTensor([self.batch_data["vocab2id"]["<unk>"]])
                    ).to(self.args["device"])
                self.pipe_data["decoderA"]["last_word"] = lastwd
                self.decode_step(k)
                prob = self.build_vocab_distribution()
                prob = torch.log(prob)

                score, wd_list = prob.data.topk(self.args["beam_size"])
                score = score.squeeze()
                wd_list = wd_list.squeeze()
                seq_base = self.beam_data[j][0]
                prob_base = self.beam_data[j][1] * float(len(seq_base))
                for i in range(self.args["beam_size"]):
                    beam_tmp.append(
                        [
                            seq_base + [wd_list[i].view(1)],
                            (prob_base + score[i]) / float(len(seq_base) + 1),
                            self.pipe_data["decoderB"],
                        ]
                    )
            beam_tmp = sorted(beam_tmp, key=lambda pb: pb[1])[::-1]
            self.beam_data = beam_tmp[: self.args["beam_size"]]

    def word_copy(self):
        """
        @author Ping Wang and Tian Shi
        Please contact ping@vt.edu or tshi@vt.edu
        Original repo from: https://github.com/wangpinggl/TREQS

        copy words from source document.
        """
        myseq = torch.cat(self.beam_data[0][0], 0)
        myattn = torch.cat(self.beam_data[0][-1]["accu_attn"], 0)
        myattn = myattn * self.batch_data["src_mask_unk"]
        beam_copy = myattn.topk(1, dim=1)[1].squeeze(-1)
        wdidx = beam_copy.data.cpu().numpy()
        out_txt = []
        myseq = torch.cat(self.beam_data[0][0], 0)
        myseq = myseq.data.cpu().numpy().tolist()
        gen_txt = [
            self.batch_data["id2vocab"][wd]
            if wd in self.batch_data["id2vocab"]
            else self.batch_data["ext_id2oov"][wd]
            for wd in myseq
        ]
        for j in range(len(gen_txt)):
            if gen_txt[j] == "<unk>":
                gen_txt[j] = self.batch_data["src_txt"][0][wdidx[j]]
        out_txt.append(" ".join(gen_txt))

        return out_txt

    def test_worker(self):
        """
        For the beam search in testing.
        """
        self.beam_search()
        try:
            myseq = self.word_copy()
        except BaseException as err:
            print(err)
            print("Running without manually word copying.")
            myseq = torch.cat(self.beam_data[0][0], 0)
            myseq = myseq.data.cpu().numpy().tolist()
            myseq = [self.batch_data["id2vocab"][idx] for idx in myseq]

        self.test_data["src_txt"] = " ".join(self.batch_data["src_txt"][0])
        self.test_data["input_sql"] = " ".join(self.batch_data["trg_txt"][0])
        self.test_data["pred_sql"] = " ".join(myseq)

    def build_batch(self, batch_id):
        """
        get batch data
        """
        output = process_batch(
            batch_id=batch_id,
            path_=self.args["results_dir"],
            fkey_=self.args["task"],
            batch_size=self.args["batch_size"],
            vocab2id=self.batch_data["vocab2id"],
            max_lens=[self.args["src_seq_len"], self.args["trg_seq_len"]],
        )

        self.batch_data["ext_id2oov"] = output["ext_id2oov"]
        self.batch_data["src_var"] = output["src_var"].to(self.args["device"])
        self.batch_data["batch_size"] = self.batch_data["src_var"].size(0)
        self.batch_data["src_seq_len"] = self.batch_data["src_var"].size(1)
        self.batch_data["src_mask_pad"] = output["src_mask_pad"].to(self.args["device"])

        if self.args["task"] == "train" or self.args["task"] == "validate":
            self.batch_data["trg_input"] = output["trg_input_var"].to(
                self.args["device"]
            )
            # different from seq2seq models.
            self.batch_data["trg_output"] = output["trg_output_var"].to(
                self.args["device"]
            )
            self.batch_data["trg_seq_len"] = self.batch_data["trg_input"].size(1)
        else:
            self.batch_data["src_mask_unk"] = output["src_mask_unk"].to(
                self.args["device"]
            )
            self.batch_data["src_txt"] = output["src_txt"]
            self.batch_data["trg_txt"] = output["trg_txt"]
            self.batch_data["trg_seq_len"] = 1

    def build_models(self):
        # Shared embedding layer
        self.train_models["embedding"] = Embedding(
            vocab_size=self.batch_data["vocab_size"],
            emb_dim=self.args["emb_dim"],
        ).to(self.args["device"])

        # Parent Encoder
        self.train_models["encoder_main"] = MainEncoder(
            self.args, self.train_models["embedding"]
        ).to(self.args["device"])

        # Parent Decoder
        # self.train_models["decoder_main"] = MainDecoder(self.args)

        self.train_models["decoderRNN"] = torch.nn.LSTMCell(
            self.args["emb_dim"] + self.args["trg_hidden_dim"],
            self.args["trg_hidden_dim"],
        ).to(self.args["device"])

        self.train_models["attnEncoder"] = AttentionEncoder(
            self.args["src_hidden_dim"],
            self.args["trg_hidden_dim"],
            attn_method="luong_general",
            repetition="temporal",
        ).to(self.args["device"])

        self.train_models["attnDecoder"] = AttentionDecoder(
            self.args["trg_hidden_dim"], attn_method="luong_general"
        ).to(self.args["device"])

        self.train_models["wrapDecoder"] = torch.nn.Linear(
            self.args["src_hidden_dim"] * 2 + self.args["trg_hidden_dim"] * 2,
            self.args["trg_hidden_dim"],
            bias=True,
        ).to(self.args["device"])

        self.train_models["genPrb"] = torch.nn.Linear(
            self.args["emb_dim"]
            + self.args["src_hidden_dim"] * 2
            + self.args["trg_hidden_dim"],
            1,
        ).to(self.args["device"])

        # Parent Decoder to vocab
        self.train_models["decoder2proj"] = torch.nn.Linear(
            self.args["trg_hidden_dim"], self.args["emb_dim"], bias=False
        ).to(self.args["device"])

    def encode(self):
        """
        Encoder Pipeline
        self.pipe_data = {
            'encoder': {},
            'decoderA': {}}
            'decoderB': {'accu_attn': [], 'last_word': word}}
        """
        src_emb, src_enc, trg_hidden0, h_attn, past_attn, past_dech = self.train_models[
            "encoder_main"
        ](self.batch_data["src_var"])

        # set up pipe_data pass to decoder
        self.pipe_data["encoder"] = {}
        self.pipe_data["encoder"]["src_emb"] = src_emb
        self.pipe_data["encoder"]["src_enc"] = src_enc

        self.pipe_data["decoderB"] = {}
        self.pipe_data["decoderB"]["hidden"] = trg_hidden0
        self.pipe_data["decoderB"]["h_attn"] = h_attn

        self.pipe_data["decoderB"]["past_attn"] = past_attn
        self.pipe_data["decoderB"]["past_dech"] = past_dech

        self.pipe_data["decoderB"]["accu_attn"] = []

        self.pipe_data["decoderFF"] = {}
        self.pipe_data["decoderFF"]["h_attn"] = []
        self.pipe_data["decoderFF"]["attn"] = []
        self.pipe_data["decoderFF"]["genPrb"] = []

        # when training get target embedding at the same time.
        if self.args["task"] == "train" or self.args["task"] == "validate":
            trg_emb = self.train_models["embedding"].get_embedding(
                self.batch_data["trg_input"]
            )
            self.pipe_data["decoderFF"]["trg_seq_emb"] = trg_emb

    def decode_step(self, k=0):
        """
        Decoder one-step t
        """
        if self.args["task"] == "train" or self.args["task"] == "validate":
            self.pipe_data["decoderA"] = self.pipe_data["decoderB"]
            word_emb = self.pipe_data["decoderFF"]["trg_seq_emb"][:, k]
        else:
            word_emb = self.train_models["embedding"].get_embedding(
                self.pipe_data["decoderA"]["last_word"]
            )

        # self.train_models["decoder_main"](self.pipe_data, word_emb)
        h_attn = self.pipe_data["decoderA"]["h_attn"]

        dec_input = torch.cat((word_emb, h_attn), 1)
        hidden = self.pipe_data["decoderA"]["hidden"]
        past_attn = self.pipe_data["decoderA"]["past_attn"]
        accu_attn = self.pipe_data["decoderA"]["accu_attn"]
        past_dech = self.pipe_data["decoderA"]["past_dech"]

        hidden = self.train_models["decoderRNN"](dec_input, hidden)
        ctx_enc, attn, attn_ee = self.train_models["attnEncoder"](
            hidden[0],
            self.pipe_data["encoder"]["src_enc"],
            past_attn,
            self.batch_data["src_mask_pad"],
        )
        # temporal attention
        past_attn = past_attn + attn_ee
        # decoder attention
        if k == 0:
            ctx_dec = Variable(
                torch.zeros(self.batch_data["batch_size"], self.args["trg_hidden_dim"])
            ).to(self.args["device"])
        else:
            ctx_dec, _ = self.train_models["attnDecoder"](hidden[0], past_dech)
        past_dech = past_dech.transpose(0, 1)  # seqL*batch*hidden
        dec_idx = past_dech.size(0)
        if k == 0:
            past_dech = hidden[0].unsqueeze(0)  # seqL*batch*hidden
            past_dech = past_dech.transpose(0, 1)  # batch*seqL*hidden
        else:
            past_dech = past_dech.contiguous().view(
                -1, self.args["trg_hidden_dim"]
            )  # seqL*batch**hidden
            past_dech = torch.cat((past_dech, hidden[0]), 0)  # (seqL+1)*batch**hidden
            past_dech = past_dech.view(
                dec_idx + 1, self.batch_data["batch_size"], self.args["trg_hidden_dim"]
            )  # (seqL+1)*batch*hidden
            past_dech = past_dech.transpose(0, 1)  # batch*(seqL+1)*hidden
        # wrap up.
        h_attn = self.train_models["wrapDecoder"](
            torch.cat((ctx_enc, ctx_dec, hidden[0]), 1)
        )
        # pointer generator
        pt_input = torch.cat((word_emb, hidden[0], ctx_enc), 1)
        genPrb = torch.sigmoid(self.train_models["genPrb"](pt_input))

        # setup piped_data
        self.pipe_data["decoderB"] = {}
        self.pipe_data["decoderB"]["h_attn"] = h_attn
        self.pipe_data["decoderB"]["past_attn"] = past_attn
        self.pipe_data["decoderB"]["hidden"] = hidden
        self.pipe_data["decoderB"]["past_dech"] = past_dech
        self.pipe_data["decoderB"]["accu_attn"] = [a for a in accu_attn]
        self.pipe_data["decoderB"]["accu_attn"].append(attn)

        if self.args["task"] == "train" or self.args["task"] == "validate":
            self.pipe_data["decoderFF"]["h_attn"].append(h_attn)
            self.pipe_data["decoderFF"]["attn"].append(attn)
            self.pipe_data["decoderFF"]["genPrb"].append(genPrb)
            if k == self.batch_data["trg_seq_len"] - 1:
                self.pipe_data["decoderFF"]["h_attn"] = (
                    torch.cat(self.pipe_data["decoderFF"]["h_attn"], 0)
                    .view(
                        self.batch_data["trg_seq_len"],
                        self.batch_data["batch_size"],
                        self.args["trg_hidden_dim"],
                    )
                    .transpose(0, 1)
                )

                self.pipe_data["decoderFF"]["attn"] = (
                    torch.cat(self.pipe_data["decoderFF"]["attn"], 0)
                    .view(
                        self.batch_data["trg_seq_len"],
                        self.batch_data["batch_size"],
                        self.args["src_seq_len"],
                    )
                    .transpose(0, 1)
                )

                self.pipe_data["decoderFF"]["genPrb"] = (
                    torch.cat(self.pipe_data["decoderFF"]["genPrb"], 0)
                    .view(self.batch_data["trg_seq_len"], self.batch_data["batch_size"])
                    .transpose(0, 1)
                )
        else:
            self.pipe_data["decoderFF"]["h_attn"] = h_attn
            self.pipe_data["decoderFF"]["attn"] = attn.unsqueeze(0)
            self.pipe_data["decoderFF"]["genPrb"] = genPrb

    def build_vocab_distribution(self):
        """
        Data flow from input to output.
        """
        trg_out = self.pipe_data["decoderFF"]["h_attn"]
        trg_out = self.train_models["decoder2proj"](trg_out)
        trg_out = self.train_models["embedding"].get_decode2vocab(trg_out)
        trg_out = trg_out.view(
            self.batch_data["batch_size"], self.batch_data["trg_seq_len"], -1
        )
        prb = torch.softmax(trg_out, dim=2)

        vocab_size = self.batch_data["vocab_size"]
        batch_size = self.batch_data["batch_size"]
        # trg_seq_len = self.batch_data['trg_seq_len']
        src_seq_len = self.batch_data["src_seq_len"]

        # pointer-generator calculate index matrix
        pt_idx = Variable(torch.FloatTensor(torch.zeros(1, 1, 1))).to(
            self.args["device"]
        )
        pt_idx = pt_idx.repeat(batch_size, src_seq_len, vocab_size)
        pt_idx.scatter_(2, self.batch_data["src_var"].unsqueeze(2), 1.0)

        p_gen = self.pipe_data["decoderFF"]["genPrb"]
        attn_ = self.pipe_data["decoderFF"]["attn"]

        prb_output = p_gen.unsqueeze(2) * prb + (1.0 - p_gen.unsqueeze(2)) * torch.bmm(
            attn_, pt_idx
        )

        return prb_output + 1e-20

    def build_pipelines(self):
        """
        Build pipeline from input to output.
        Output is loss.
        Input is word one-hot encoding.
        """
        self.encode()
        for k in range(self.args["trg_seq_len"]):
            self.decode_step(k)
        pred_output = self.build_vocab_distribution()

        pad_mask = torch.ones(self.batch_data["vocab_size"]).to(self.args["device"])
        pad_mask[self.batch_data["vocab2id"]["<pad>"]] = 0
        self.loss_criterion = torch.nn.NLLLoss(pad_mask).to(self.args["device"])

        pred_output = torch.log(pred_output)
        pred_output = pred_output.reshape(-1, self.batch_data["vocab_size"])
        target_output = self.batch_data["trg_output"].view(-1)

        loss = self.loss_criterion(
            pred_output,
            target_output,
        )

        return loss

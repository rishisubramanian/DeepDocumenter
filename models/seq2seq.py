import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model, register_model_architecture

class Seq2SeqEncoder(FairseqEncoder):

    def __init__(
        self, args, dictionary, embed_dim=128, hidden_dim=128, dropout=0.1,
    ):
        super().__init__(dictionary)
        self.args = args

        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )
        self.dropout = nn.Dropout(p=dropout)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
        )

    def forward(self, src_tokens, src_lengths):
        if self.args.left_pad_source:
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                padding_idx=self.dictionary.pad(),
                left_to_right=True
            )

        x = self.embed_tokens(src_tokens)

        x = self.dropout(x)

        x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True)

        _, final_hidden = self.gru(x)

        return {
            'final_hidden': final_hidden.squeeze(0),
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        final_hidden = encoder_out['final_hidden']
        return {
            'final_hidden': final_hidden.index_select(0, new_order),
        }

class Seq2SeqDecoder(FairseqIncrementalDecoder):

    def __init__(
        self, dictionary, encoder_hidden_dim=128, embed_dim=128, hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__(dictionary)
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(
            input_size=encoder_hidden_dim + embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
        )
        self.output_projection = nn.Linear(hidden_dim, len(dictionary))

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, tgt_len = prev_output_tokens.size()
        final_encoder_hidden = encoder_out['final_hidden']
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout(x)
        x = torch.cat(
            [x, final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)],
            dim=2,
        )

        initial_state = utils.get_incremental_state(
            self, incremental_state, 'prev_state',
        )
        if initial_state is None:
            initial_state = (
                final_encoder_hidden.unsqueeze(0),
                torch.zeros_like(final_encoder_hidden).unsqueeze(0),
            )

        output, latest_state = self.gru(x.transpose(0, 1), initial_state)

        utils.set_incremental_state(
            self, incremental_state, 'prev_state', latest_state,
        )

        x = output.transpose(0, 1)
        x = self.output_projection(x)
        return x, None

    def reorder_incremental_state(self, incremental_state, new_order):
        prev_state = utils.get_incremental_state(
            self, incremental_state, 'prev_state',
        )

        reordered_state = (
            prev_state[0].index_select(1, new_order),
            prev_state[1].index_select(1, new_order),
        )

        utils.set_incremental_state(
            self, incremental_state, 'prev_state', reordered_state,
        )

@register_model('my_seq2seq')
class Seq2SeqModel(FairseqModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--encoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the encoder embeddings',
        )
        parser.add_argument(
            '--encoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the encoder hidden state',
        )
        parser.add_argument(
            '--encoder-dropout', type=float, default=0.1,
            help='encoder dropout probability',
        )
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )
        parser.add_argument(
            '--decoder-dropout', type=float, default=0.1,
            help='decoder dropout probability',
        )

    @classmethod
    def build_model(cls, args, task):
        encoder = Seq2SeqEncoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_dim=args.encoder_hidden_dim,
            dropout=args.encoder_dropout,
        )
        decoder = Seq2SeqDecoder(
            dictionary=task.target_dictionary,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_dim=args.decoder_hidden_dim,
            dropout=args.decoder_dropout,
        )
        model = Seq2SeqModel(encoder, decoder)

        print(model)

        return model


@register_model_architecture('my_seq2seq', 'gru_seq2seq')
def my_seq2seq(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 128)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 128)
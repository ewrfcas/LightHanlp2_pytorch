import torch
import torch.nn as nn
from light_hanlp.utils.preprocess import get_dep_sdp_inputs, get_dep_outputs, get_sdp_outputs


class MLP(nn.Module):
    def __init__(self, input_hidden, output_hidden):
        super(MLP, self).__init__()
        self.dense = nn.Linear(input_hidden, output_hidden)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        return x


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out, bias_x, bias_y):
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.zeros(self.n_out,
                                               self.n_in + self.bias_x,
                                               self.n_in + self.bias_y))

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, torch.ones_like(x[:, :, 1]).unsqueeze(-1)], dim=-1)
        if self.bias_y:
            y = torch.cat([y, torch.ones_like(y[:, :, 1]).unsqueeze(-1)], dim=-1)

        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        if self.n_out == 1:
            s = s.squeeze(1)
        return s


class BiaffineModel(nn.Module):
    def __init__(self, config, vocab):
        super(BiaffineModel, self).__init__()
        self.n_rels = config['n_rels']
        self.n_feats = config['n_feats']
        self.n_words = config['n_words']
        self.embedding_dim = config['n_embed']
        self.n_mlp_rel = config['n_mlp_rel']
        self.n_mlp_arc = config['n_mlp_arc']
        self.n_lstm_layers = config['n_lstm_layers']
        self.n_lstm_hidden = config['n_lstm_hidden']
        self.n_ext_words = len(vocab['form_vocab']['idx_to_token'])
        self.n_words = config['n_words']
        self.token_to_idx = {w: i for i, w in enumerate(vocab['form_vocab']['idx_to_token'])}
        self.pos_to_idx = {w: i for i, w in enumerate(vocab['cpos_vocab']['idx_to_token'])}
        self.rel_vocab = {i: w for i, w in enumerate(vocab['rel_vocab']['idx_to_token'])}

        self.word_embed = nn.Embedding(self.n_words, self.embedding_dim, padding_idx=0)
        self.feat_embed = nn.Embedding(self.n_feats, self.embedding_dim, padding_idx=0)
        self.ext_word_embed = nn.Embedding(self.n_ext_words, self.embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(self.embedding_dim * 2, self.n_lstm_hidden, bidirectional=True,
                            batch_first=True, num_layers=self.n_lstm_layers)

        self.mlp_arc_h = MLP(self.n_lstm_hidden * 2, self.n_mlp_arc)
        self.mlp_arc_d = MLP(self.n_lstm_hidden * 2, self.n_mlp_arc)
        self.mlp_rel_h = MLP(self.n_lstm_hidden * 2, self.n_mlp_rel)
        self.mlp_rel_d = MLP(self.n_lstm_hidden * 2, self.n_mlp_rel)

        self.arc_attn = Biaffine(n_in=self.n_mlp_arc,
                                 n_out=1,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=self.n_mlp_rel,
                                 n_out=self.n_rels,
                                 bias_x=True,
                                 bias_y=True)

    def forward(self, input_ids, ext_input_ids, pos_ids, mask, x_lengths):
        word_embed = self.word_embed(input_ids)
        word_embed += self.ext_word_embed(ext_input_ids)
        pos_embed = self.feat_embed(pos_ids)
        embed = torch.cat([word_embed, pos_embed], dim=-1)

        x = torch.nn.utils.rnn.pack_padded_sequence(embed, x_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)
        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h)
        s_rel = s_rel.permute([0, 2, 3, 1])
        # set the scores that exceed the length of each sentence to -inf
        # [bs, len]->[bs, 1, len]
        mask_ = mask.unsqueeze(1).repeat([1, mask.shape[1], 1]).to(dtype=s_arc.dtype)
        mask_ = (1 - mask_) * -50000
        s_arc = s_arc + mask_

        return s_arc, s_rel

    def predict(self, inputs, task, device='cpu'):
        input_ids, ext_input_ids, pos_ids, mask, x_lengths = get_dep_sdp_inputs(inputs, self.token_to_idx,
                                                                                self.pos_to_idx,
                                                                                word_embed_range=self.n_words)
        input_ids_tensor = torch.tensor(input_ids).to(device)
        ext_input_ids_tensor = torch.tensor(ext_input_ids).to(device)
        pos_ids_tensor = torch.tensor(pos_ids).to(device)
        mask_tensor = torch.tensor(mask).to(device)
        x_lengths_tensor = torch.tensor(x_lengths).to(device)
        with torch.no_grad():
            arc_scores, rel_scores = self.forward(input_ids=input_ids_tensor,
                                                  ext_input_ids=ext_input_ids_tensor,
                                                  pos_ids=pos_ids_tensor,
                                                  mask=mask_tensor,
                                                  x_lengths=x_lengths_tensor)

        arc_scores = arc_scores.cpu().numpy()
        rel_scores = rel_scores.cpu().numpy()
        if task == 'dep':
            results = get_dep_outputs(arc_scores, rel_scores, x_lengths, self.rel_vocab)
        elif task == 'sdp':
            results = get_sdp_outputs(arc_scores, rel_scores, x_lengths, self.rel_vocab)
        else:
            raise NotImplementedError

        return results

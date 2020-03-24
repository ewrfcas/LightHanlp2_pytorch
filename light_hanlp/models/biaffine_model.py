import torch
import torch.nn as nn
import torch.functional as F


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

        x = torch.nn.utils.rnn.pack_padded_sequence(embed, x_lengths, batch_first=True)
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


# import json
#
# config = json.load(open('../pytorch_models/dep/biaffine_ctb7_20200109_022431/config.json'))
# vocabs = json.load(open('../pytorch_models/dep/biaffine_ctb7_20200109_022431/vocabs.json'))
# model = BiaffineModel(config, vocabs)
#
#
# def torch_init_model(model, init_checkpoint):
#     state_dict = torch.load(init_checkpoint, map_location='cpu')
#     missing_keys = []
#     unexpected_keys = []
#     error_msgs = []
#     # copy state_dict so _load_from_state_dict can modify it
#     metadata = getattr(state_dict, '_metadata', None)
#     state_dict = state_dict.copy()
#     if metadata is not None:
#         state_dict._metadata = metadata
#
#     def load(module, prefix=''):
#         local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
#
#         module._load_from_state_dict(
#             state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
#         for name, child in module._modules.items():
#             if child is not None:
#                 load(child, prefix + name + '.')
#
#     load(model, prefix='')
#
#     print("missing keys:{}".format(missing_keys))
#     print('unexpected keys:{}'.format(unexpected_keys))
#     print('error msgs:{}'.format(error_msgs))
#
#
# torch_init_model(model, '../pytorch_models/dep/biaffine_ctb7_20200109_022431/model.pth')
#
# import numpy as np
#
#
# def input_padding(input_ids, pad_id):
#     max_len = max([len(li) for li in input_ids])
#     for input_ids_ in input_ids:
#         if len(input_ids_) < max_len:
#             input_ids_.extend([pad_id] * (max_len - len(input_ids_)))
#     return input_ids
#
#
# def X_to_inputs(sents, token_to_idx, pos_to_idx, word_embed_range, root_tok='<bos>',
#                 root_pos='<bos>', unk_tok='<unk>', pad_tok='<pad>', unk_pos='<bos>', pad_pos='<bos>'):
#     if type(sents) == str:
#         sents = [sents]
#     ext_input_ids = []
#     input_ids = []
#     pos_ids = []
#     mask = []
#     x_lengths = []
#     for sent in sents:
#         # 句法分析开头先加个root
#         ext_input_ids.append([token_to_idx[root_tok]])
#         input_ids.append([token_to_idx[root_tok]])
#         pos_ids.append([pos_to_idx[root_pos]])
#         mask.append([1])
#         for word, pos in sent:
#             if word in token_to_idx:
#                 if token_to_idx[word] < word_embed_range:
#                     input_ids[-1].append(token_to_idx[word])
#                 else:
#                     input_ids[-1].append(token_to_idx[unk_tok])
#                 ext_input_ids[-1].append(token_to_idx[word])
#             else:
#                 input_ids[-1].append(token_to_idx[unk_tok])
#                 ext_input_ids[-1].append(token_to_idx[unk_tok])
#             if pos in pos_to_idx:
#                 pos_ids[-1].append(pos_to_idx[pos])
#             else:
#                 pos_ids[-1].append(pos_to_idx[unk_pos])
#
#             mask[-1].append(1)
#
#         x_lengths.append(len(ext_input_ids[-1]))
#
#     if len(input_ids) > 1:
#         input_ids = input_padding(input_ids, token_to_idx[pad_tok])
#         ext_input_ids = input_padding(ext_input_ids, token_to_idx[pad_tok])
#         pos_ids = input_padding(pos_ids, token_to_idx[pad_pos])
#         mask = input_padding(mask, 0)
#
#     return np.array(input_ids), np.array(ext_input_ids), np.array(pos_ids), np.array(mask), np.array(x_lengths)
#
#
# inputs = [[('蜡烛', 'NN'), ('两', 'CD'), ('头', 'NN'), ('烧', 'VV')]]
# token_to_idx = {w: i for i, w in enumerate(vocabs['form_vocab']['idx_to_token'])}
# pos_to_idx = {w: i for i, w in enumerate(vocabs['cpos_vocab']['idx_to_token'])}
# rel_vocab = {i: w for i, w in enumerate(vocabs['rel_vocab']['idx_to_token'])}
# input_ids, ext_input_ids, pos_ids, mask, x_lengths = X_to_inputs(inputs, token_to_idx, pos_to_idx,
#                                                                  word_embed_range=config['n_words'])
#
# model.eval()
# with torch.no_grad():
#     arc_scores, rel_scores = model(input_ids=torch.tensor(input_ids),
#                                    ext_input_ids=torch.tensor(ext_input_ids),
#                                    pos_ids=torch.tensor(pos_ids),
#                                    mask=torch.tensor(mask),
#                                    x_lengths=torch.tensor(x_lengths))
# arc_scores = arc_scores.cpu().numpy()
# rel_scores = rel_scores.cpu().numpy()
#
#
# def Y_to_outputs(arc_scores, rel_scores, lengths, rel_vocab):
#     sents = []
#     arc_preds = np.argmax(arc_scores, -1)
#     rel_preds = np.argmax(rel_scores, -1)
#
#     for arc_sent, rel_sent, length in zip(arc_preds, rel_preds, lengths):
#         arcs = list(arc_sent)[1:length + 1]
#         rels = list(rel_sent)[1:length + 1]
#         sents.append([(a, rel_vocab[r[a]]) for a, r in zip(arcs, rels)])
#
#     return sents
#
# print(inputs)
# print(Y_to_outputs(arc_scores, rel_scores, x_lengths, rel_vocab))

import torch
import torch.nn as nn
import torch.functional as F


class WeightNormConv1D(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3):
        super(WeightNormConv1D, self).__init__()
        self.conv1d = nn.Conv1d(input_channel, output_channel, kernel_size, padding=1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        return torch.transpose(self.conv1d(x), 1, 2)


class ConvTagger(nn.Module):
    def __init__(self, config, vocab, embedding_dim=100):
        super(ConvTagger, self).__init__()
        self.filters = config['filters']
        self.class_num = len(vocab['tag_vocab']['idx_to_token'])
        self.embeddings = nn.Embedding(len(vocab['word_vocab']['idx_to_token']), embedding_dim)
        self.Conv1Dv = nn.ModuleList()
        self.Conv1Dw = nn.ModuleList()
        for i, f in enumerate(self.filters):
            if i == 0:
                self.Conv1Dv.append(WeightNormConv1D(embedding_dim, f))
                self.Conv1Dw.append(WeightNormConv1D(embedding_dim, f))
            else:
                self.Conv1Dv.append(WeightNormConv1D(f, f))
                self.Conv1Dw.append(WeightNormConv1D(f, f))

        self.dense = nn.Linear(self.filters[-1], self.class_num, bias=False)

    def forward(self, x):
        x = self.embeddings(x)
        for conv1dv, conv1dw in zip(self.Conv1Dv, self.Conv1Dw):
            xw = conv1dw(x)
            xv = conv1dv(x)
            x = xw * torch.sigmoid(xv)
        logits = self.dense(x)
        return logits


import json

config = json.load(open('../pytorch_models/cws/pku98_6m_conv_ngram_20200110_134736/config.json'))
vocabs = json.load(open('../pytorch_models/cws/pku98_6m_conv_ngram_20200110_134736/vocabs.json'))
model = ConvTagger(config, vocabs)


def torch_init_model(model, init_checkpoint):
    state_dict = torch.load(init_checkpoint, map_location='cpu')
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))


torch_init_model(model, '../pytorch_models/cws/pku98_6m_conv_ngram_20200110_134736/model.pth')

import numpy as np

def input_padding(input_ids, pad_id):
    max_len = max([len(li) for li in input_ids])
    for input_ids_ in input_ids:
        if len(input_ids_) < max_len:
            input_ids_.extend([pad_id] * (max_len - len(input_ids_)))
    return input_ids


def X_to_inputs(sents, word_vocab, token_to_idx):
    if type(sents) == str:
        sents = [sents]
    input_ids = []
    unk = word_vocab['unk_token']
    pad_id = token_to_idx[word_vocab['pad_token']]
    for sent in sents:
        input_ids.append([])
        for word in sent:
            if word in token_to_idx:
                input_ids[-1].append(token_to_idx[word])
            else:
                input_ids[-1].append(token_to_idx[unk])
    if len(input_ids) > 1:
        input_ids = input_padding(input_ids, pad_id)
    return np.array(input_ids)


token_to_idx = {w: i for i, w in enumerate(vocabs['word_vocab']['idx_to_token'])}
input_ids = X_to_inputs('萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。', vocabs['word_vocab'], token_to_idx)
print(input_ids)
with torch.no_grad():
    output = model(torch.tensor(input_ids))
output = output.cpu().numpy()
print(output)


def bmes_to_words(chars, tags):
    result = []
    if len(chars) == 0:
        return result
    word = chars[0]

    for c, t in zip(chars[1:], tags[1:]):
        if t == 'B' or t == 'S':
            result.append(word)
            word = ''
        word += c
    if len(word) != 0:
        result.append(word)

    return result


def Y_to_tokens(tag_vocab, Y, inputs):
    Y = np.argmax(Y, axis=2)
    results = []
    for y, inp in zip(Y, inputs):
        tags = [tag_vocab[int(y_)] for y_ in y[:len(inp)]]
        results.append(bmes_to_words(list(inp), tags))
    return results


tag_vocab = {i: w for i, w in enumerate(vocabs['tag_vocab']['idx_to_token'])}
print(Y_to_tokens(tag_vocab, output, ['萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。']))

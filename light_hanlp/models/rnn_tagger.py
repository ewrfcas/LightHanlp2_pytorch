import torch
import torch.nn as nn
import fasttext
import torch.functional as F


class RnnTagger(nn.Module):
    def __init__(self, config, vocab, fast_text_path=None, embedding_dim=300):
        super(RnnTagger, self).__init__()
        self.class_num = len(vocab['tag_vocab']['idx_to_token'])
        self.fast_text_embedding = fasttext.load_model(fast_text_path)
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=config['rnn_units'], batch_first=True,
                              bidirectional=True)
        self.dense = nn.Linear(config['rnn_units'] * 2, self.class_num)

    def embed(self, words):
        tensors = []
        for word in words:
            tensors.append(self.fast_text_embedding[word])
        return tensors

    def get_embedding(self, input_tokens):
        embed_output = []
        lengths = []
        for sentence in input_tokens:
            embed_output.append(self.embed(sentence))
            lengths.append(len(sentence))
        max_len = max([len(e) for e in embed_output])
        for embs in embed_output:
            if len(embs) < max_len:
                embs.extend([self.fast_text_embedding['<pad>']] * (max_len - len(embs)))
        return torch.tensor(embed_output), torch.tensor(lengths)

    def forward(self, x):
        embed_output, lengths = self.get_embedding(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(embed_output, lengths, batch_first=True)
        x, _ = self.bilstm(x)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        logits = self.dense(output)
        return logits


import json

config = json.load(open('../pytorch_models/pos/ctb5_pos_rnn_fasttext_20191230_202639/config.json'))
vocabs = json.load(open('../pytorch_models/pos/ctb5_pos_rnn_fasttext_20191230_202639/vocabs.json'))
model = RnnTagger(config, vocabs, fast_text_path='../pytorch_models/third/wiki.zh/wiki.zh.bin')

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


torch_init_model(model, '../pytorch_models/pos/ctb5_pos_rnn_fasttext_20191230_202639/model.pth')

tag_vocab = {i: w for i, w in enumerate(vocabs['tag_vocab']['idx_to_token'])}
import numpy as np


def Y_to_tokens(tag_vocab, Y, inputs):
    Y = np.argmax(Y, axis=2)
    results = []
    for y, inp in zip(Y, inputs):
        tags = [tag_vocab[int(y_)] for y_ in y[:len(inp)]]
        results.append(tags)
    return results


inputs = [
    ["HanLP", "是", "一", "系列", "模型", "与", "算法", "组成", "的", "自然", "语言", "处理", "工具包", "，", "目标", "是", "普及", "自然", "语言", "处理", "在", "生产", "环境", "中", "的", "应用", "。"],
    ["HanLP", "具备", "功能", "完善", "、", "性能", "高效", "、", "架构", "清晰", "、", "语料", "时", "新", "、", "可", "自", "定义", "的", "特点", "。"],
    ["内部", "算法", "经过", "工业界", "和", "学术界", "考验", "，", "配套", "书籍", "《", "自然", "语言", "处理", "入门", "》", "已经", "出版", "。"]
  ]

# inputs = [['我', '的', '希望', '是', '希望', '和平']]
with torch.no_grad():
    logits = model(inputs)

logits = logits.cpu().numpy()
print(Y_to_tokens(tag_vocab, logits, inputs))

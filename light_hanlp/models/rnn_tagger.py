import torch
import torch.nn as nn
import fasttext
from light_hanlp.utils.char_table import CharTable
from light_hanlp.utils.preprocess import get_pos_outputs


class RnnTagger(nn.Module):
    def __init__(self, config, vocab, fast_text_path=None, embedding_dim=300):
        super(RnnTagger, self).__init__()
        self.class_num = len(vocab['tag_vocab']['idx_to_token'])
        self.tag_vocab = {i: w for i, w in enumerate(vocab['tag_vocab']['idx_to_token'])}
        self.fast_text_embedding = fasttext.load_model(fast_text_path)
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=config['rnn_units'], batch_first=True,
                              bidirectional=True)
        self.dense = nn.Linear(config['rnn_units'] * 2, self.class_num)

    def embed(self, words):
        tensors = []
        for word in words:
            word_ = CharTable.normalize_text(word)
            tensors.append(self.fast_text_embedding[word_])
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

    def forward(self, embed_output, lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(embed_output, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.bilstm(x)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        logits = self.dense(output)
        return logits

    def predict(self, inputs, device='cpu'):
        if type(inputs[0]) == str:
            inputs = [inputs]
        embed_output, lengths = self.get_embedding(inputs)
        embed_output = embed_output.to(device)
        lengths = lengths.to(device)
        with torch.no_grad():
            logits = self.forward(embed_output, lengths)
        logits = logits.cpu().numpy()
        results = get_pos_outputs(self.tag_vocab, logits, inputs)

        return results

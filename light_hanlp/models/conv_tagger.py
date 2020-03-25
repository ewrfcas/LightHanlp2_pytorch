import torch
import torch.nn as nn
import torch.nn.functional as F
from light_hanlp.utils.preprocess import get_cws_inputs, get_cws_outputs


class WeightNormConv1D(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3):
        super(WeightNormConv1D, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(output_channel, input_channel, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(output_channel))
        self.g = nn.Parameter(torch.Tensor(output_channel))


    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        v = self.weight / torch.functional.norm(self.weight, dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
        weight_norm = self.g.unsqueeze(-1).unsqueeze(-1) * v
        x = F.conv1d(x, weight_norm, self.bias, 1, 1, 1, 1)

        return torch.transpose(x, 1, 2)


class ConvTagger(nn.Module):
    def __init__(self, config, vocab, embedding_dim=100):
        super(ConvTagger, self).__init__()
        self.filters = config['filters']
        self.unk_tok = vocab['word_vocab']['unk_token']
        self.pad_tok = vocab['word_vocab']['pad_token']
        self.class_num = len(vocab['tag_vocab']['idx_to_token'])
        self.tag_vocab = {i: w for i, w in enumerate(vocab['tag_vocab']['idx_to_token'])}
        self.token_to_idx = {w: i for i, w in enumerate(vocab['word_vocab']['idx_to_token'])}
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

    def predict(self, sents, device='cpu'):
        if type(sents) == str:
            sents = [sents]
        input_ids = get_cws_inputs(sents, self.token_to_idx, self.unk_tok, self.pad_tok)
        input_ids = torch.tensor(input_ids).to(device)
        with torch.no_grad():
            output = self.forward(input_ids)
        output = output.cpu().numpy()
        output = get_cws_outputs(self.tag_vocab, output, sents)
        return output

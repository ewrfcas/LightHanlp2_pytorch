import h5py
import torch

# 要存储pytorch权重的字典
pytorch_dict = {}


# 遍历keras的权重参数
def get_groups(f, fs, gs):
    for g in fs.keys():
        f2 = f['/'.join(gs + [g])]
        if type(f2) != h5py._hl.group.Group:
            print(f2.name, f2.shape)
        else:
            gs2 = list(gs + [g])
            get_groups(f, f2, gs2)


# 转化dep-baffine模型
def get_dep_baffine_groups(f, fs, gs):
    for g in fs.keys():
        f2 = f['/'.join(gs + [g])]
        if type(f2) != h5py._hl.group.Group:
            weights = torch.tensor(f2.value)
            names = gs + [g]

            if names[0] == 'arc_attn':
                names = 'arc_attn.weight'
            elif names[0] == 'rel_attn':
                names = 'rel_attn.weight'
            elif names[0] == 'glove.6B.100d':
                names = 'ext_word_embed.weight'
            elif names[0] == 'feat_embed':
                names = 'feat_embed.weight'
            elif names[0] == 'word_embed':
                names = 'word_embed.weight'
            elif 'mlp' in names[0]:
                names = [names[0], 'dense', names[-1]]
                names[-1] = names[-1].replace(':0', '').replace('kernel', 'weight')
                names = '.'.join(names)
            elif names[0] == 'lstm':
                if names[-1] == 'kernel:0':
                    wn = 'weight_ih_l'
                elif names[-1] == 'recurrent_kernel:0':
                    wn = 'weight_hh_l'
                elif names[-1] == 'bias:0':
                    wn = 'bias_ih_l'

                n_layer = names[3].split('_')
                if len(n_layer) == 1:
                    n_layer = '0'
                else:
                    n_layer = n_layer[1]
                wn += n_layer

                if 'backward_' in names[4]:
                    wn += '_reverse'
                names = '.'.join([names[0], wn])

                # pytorch的lstm在hh多一个bias，这里额外附加全0bias
                if 'bias' in names:
                    b_name = names.replace('bias_ih', 'bias_hh')
                    b_weights = torch.zeros_like(weights)
                    print(b_name, b_weights.shape)
                    pytorch_dict[b_name] = b_weights

            else:
                names = '.'.join(names)
            if 'weight' in names and ('lstm' in names or 'dense' in names):
                weights = torch.transpose(weights, 0, 1)
            print(names, weights.shape)
            pytorch_dict[names] = weights


        else:
            gs2 = list(gs + [g])
            get_dep_baffine_groups(f, f2, gs2)


# 转化sdp-baffine模型，我也搞不懂为什么作者sdp的baffine模型参数命名那么乱，可能是clone出来的
def get_sdp_baffine_groups(f, fs, gs):
    for g in fs.keys():
        f2 = f['/'.join(gs + [g])]
        if type(f2) != h5py._hl.group.Group:
            weights = torch.tensor(f2.value)
            names = gs + [g]

            if names[0] == 'keras_biaffine':
                names = 'arc_attn.weight'
            elif names[0] == 'keras_biaffine_1':
                names = 'rel_attn.weight'
            elif names[0] == 'embedding_2':
                names = 'ext_word_embed.weight'
            elif names[0] == 'embedding_1':
                names = 'feat_embed.weight'
            elif names[0] == 'embedding':
                names = 'word_embed.weight'
            elif 'keras_mlp' in names[0]:
                if names[0] == 'keras_mlp':
                    names[0] = 'mlp_arc_h'
                elif names[0] == 'keras_mlp_1':
                    names[0] = 'mlp_arc_d'
                elif names[0] == 'keras_mlp_2':
                    names[0] = 'mlp_rel_h'
                elif names[0] == 'keras_mlp_3':
                    names[0] = 'mlp_rel_d'
                names = [names[0], 'dense', names[-1]]
                names[-1] = names[-1].replace(':0', '').replace('kernel', 'weight')
                names = '.'.join(names)
            elif 'lstm' in names[4]:
                names[0] = 'lstm'
                if names[-1] == 'kernel:0':
                    wn = 'weight_ih_l'
                elif names[-1] == 'recurrent_kernel:0':
                    wn = 'weight_hh_l'
                elif names[-1] == 'bias:0':
                    wn = 'bias_ih_l'

                n_layer = names[3].split('_')
                if len(n_layer) == 1:
                    n_layer = '0'
                else:
                    n_layer = n_layer[1]
                wn += n_layer

                if 'backward_' in names[4]:
                    wn += '_reverse'
                names = '.'.join([names[0], wn])

                # pytorch的lstm在hh多一个bias，这里额外附加全0bias
                if 'bias' in names:
                    b_name = names.replace('bias_ih', 'bias_hh')
                    b_weights = torch.zeros_like(weights)
                    print(b_name, b_weights.shape)
                    pytorch_dict[b_name] = b_weights

            else:
                names = '.'.join(names)
            if 'weight' in names and ('lstm' in names or 'dense' in names):
                weights = torch.transpose(weights, 0, 1)
            print(names, weights.shape)
            pytorch_dict[names] = weights


        else:
            gs2 = list(gs + [g])
            get_sdp_baffine_groups(f, f2, gs2)


# 转化bert模型
def get_bert_groups(f, fs, gs):
    for g in fs.keys():
        f2 = f['/'.join(gs + [g])]
        if type(f2) != h5py._hl.group.Group:
            weights = torch.tensor(f2.value)
            names = gs + [g]
            if names[0] == names[1]:
                names = names[1:]
            if names[-1] == 'embeddings:0':
                names[-1] = 'weight'
            if names[-1] == 'kernel:0':
                weights = torch.transpose(weights, 0, 1)
            names = '.'.join(names)
            names = names.replace('layer_', 'layer.').replace('beta', 'bias').replace('gamma', 'weight') \
                .replace(':0', '').replace('kernel', 'weight').replace('intermediate.', 'intermediate.dense.')
            pytorch_dict[names] = weights
            print('Convert Success:', names, weights.shape)
        else:
            gs2 = list(gs + [g])
            get_bert_groups(f, f2, gs2)


# 转化rnn_tagger模型
def get_rnn_tagger_groups(f, fs, gs):
    for g in fs.keys():
        f2 = f['/'.join(gs + [g])]
        if type(f2) != h5py._hl.group.Group:
            weights = torch.tensor(f2.value)
            names = gs + [g]
            if 'lstm' in names[-2]:
                is_reverse = True if 'backward' in names[-2] else False
                if names[-1] == 'kernel:0':
                    names[-2] = 'weight_ih_l0'

                elif names[-1] == 'recurrent_kernel:0':
                    names[-2] = 'weight_hh_l0'

                elif names[-1] == 'bias:0':
                    names[-2] = 'bias_ih_l0'

                if is_reverse:
                    names[-2] += '_reverse'
                names = names[:-1]

            if names[-1] == 'kernel:0':
                names[-1] = 'weight'

            if names[-1] == 'bias:0':
                names[-1] = 'bias'

            if names[0] == names[1]:
                names = names[1:]
            names = '.'.join(names)
            if 'weight' in names:
                weights = torch.transpose(weights, 0, 1)

            if 'bias_' in names:
                # pytorch的lstm在hh多一个bias，这里额外附加全0bias
                b_name = names.replace('bias_ih', 'bias_hh')
                b_weights = torch.zeros_like(weights)
                print('Convert Success:', b_name, b_weights.shape)
                pytorch_dict[b_name] = b_weights

            pytorch_dict[names] = weights
            print('Convert Success:', names, weights.shape)
        else:
            gs2 = list(gs + [g])
            get_rnn_tagger_groups(f, f2, gs2)


# 转化cnn_tagger模型
def get_cnn_tagger_groups(f, fs, gs):
    for g in fs.keys():
        f2 = f['/'.join(gs + [g])]
        if type(f2) != h5py._hl.group.Group:
            weights = torch.tensor(f2.value)
            names = gs + [g]
            if names[0] == 'character.vec':
                names = 'embeddings.weight'
            elif names[0] == 'dense':
                names = 'dense.weight'
            elif 'Conv' in names[0]:
                if 'initialized' in names[-1]:
                    continue
                nlayer = int(names[0].split('_')[1])
                names[0] = names[0].split('_')[0]
                names[1] = str(nlayer)
                names[2] = names[2].replace(':0', '').replace('kernel', 'weight')
                if names[2] == 'g':
                    continue
                names = '.'.join(names)
            if 'Conv1D' in names and 'weight' in names:
                weights = torch.transpose(weights, 0, 2)
            if 'dense' in names and 'weight' in names:
                weights = torch.transpose(weights, 0, 1)

            pytorch_dict[names] = weights
            print('Convert Success:', names, weights.shape)
        else:
            gs2 = list(gs + [g])
            get_cnn_tagger_groups(f, f2, gs2)


if __name__ == '__main__':
    f = h5py.File('../../models/pos/ctb5_pos_rnn_fasttext_20191230_202639/model.h5', mode='a')

    print('Show keras parameters...')
    for g in f.keys():
        get_groups(f, f[g], [g])

    print('Start Converting...')
    for g in f.keys():
        get_rnn_tagger_groups(f, f[g], [g])

    # torch.save(pytorch_dict, '../../pytorch_models/pos/ctb5_pos_rnn_fasttext_20191230_202639/model.pth')

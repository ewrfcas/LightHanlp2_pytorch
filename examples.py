from light_hanlp.models.conv_tagger import ConvTagger
from light_hanlp.models.rnn_tagger import RnnTagger
from light_hanlp.models.bert_model import BertForTokenClassification, BertConfig
from light_hanlp.models.biaffine_model import BiaffineModel
from light_hanlp.utils.utils import torch_init_model
import json

if __name__ == '__main__':
    inputs = [
        "HanLP是一系列模型与算法组成的自然语言处理工具包，目标是普及自然语言处理在生产环境中的应用。",
        "HanLP具备功能完善、性能高效、架构清晰、语料时新、可自定义的特点。",
        "内部算法经过工业界和学术界考验，配套书籍《自然语言处理入门》已经出版。",
        "上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。",
        "萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。",
        "HanLP支援臺灣正體、香港繁體，具有新詞辨識能力的中文斷詞系統",
    ]
    print('inputs:')
    for inp in inputs:
        print(inp)
    print('#' * 100)
    print('\n')

    # 加载分词模型
    print('加载CWS模型...')
    config = json.load(open('pytorch_models/cws/pku98_6m_conv_ngram_20200110_134736/config.json'))
    vocabs = json.load(open('pytorch_models/cws/pku98_6m_conv_ngram_20200110_134736/vocabs.json'))
    cws_model = ConvTagger(config, vocabs)
    torch_init_model(cws_model, 'pytorch_models/cws/pku98_6m_conv_ngram_20200110_134736/model.pth')
    cws_model.eval()
    cws_results = cws_model.predict(inputs)
    for result in cws_results:
        print(result)
    print('#' * 100)
    print('\n')

    # 加载POS模型
    print('加载POS模型...')
    config = json.load(open('pytorch_models/pos/ctb5_pos_rnn_fasttext_20191230_202639/config.json'))
    vocabs = json.load(open('pytorch_models/pos/ctb5_pos_rnn_fasttext_20191230_202639/vocabs.json'))
    pos_model = RnnTagger(config, vocabs, fast_text_path='pytorch_models/third/wiki.zh/wiki.zh.bin')
    torch_init_model(pos_model, 'pytorch_models/pos/ctb5_pos_rnn_fasttext_20191230_202639/model.pth')
    pos_model.eval()
    pos_results = pos_model.predict(cws_results)
    for result in pos_results:
        print(result)
    print('#' * 100)
    print('\n')

    # 加载NER模型
    print('加载NER模型...')
    config = json.load(open('pytorch_models/ner/ner_bert_base_msra_20200104_185735/config.json'))
    vocabs = json.load(open('pytorch_models/ner/ner_bert_base_msra_20200104_185735/vocabs.json'))
    bert_config = BertConfig.from_json_file('pytorch_models/ner/ner_bert_base_msra_20200104_185735/bert_config.json')
    ner_model = BertForTokenClassification(bert_config,
                                           config,
                                           vocabs,
                                           vocab_file='pytorch_models/ner/ner_bert_base_msra_20200104_185735/vocab.txt',
                                           num_labels=len(vocabs['tag_vocab']['idx_to_token']))
    torch_init_model(ner_model, 'pytorch_models/ner/ner_bert_base_msra_20200104_185735/model.pth')
    ner_model.eval()
    ner_results = ner_model.predict(inputs)
    for result in ner_results:
        print(result)
    print('#' * 100)
    print('\n')

    # 加载DEP模型
    print('加载DEP模型...')
    config = json.load(open('pytorch_models/dep/biaffine_ctb7_20200109_022431/config.json'))
    vocabs = json.load(open('pytorch_models/dep/biaffine_ctb7_20200109_022431/vocabs.json'))
    dep_model = BiaffineModel(config, vocabs)
    torch_init_model(dep_model, 'pytorch_models/dep/biaffine_ctb7_20200109_022431/model.pth')
    dep_model.eval()
    tok_pos_inputs = []
    for tokens, poses in zip(cws_results, pos_results):
        tok_pos_input = []
        for tok, pos in zip(tokens, poses):
            tok_pos_input.append((tok, pos))
        tok_pos_inputs.append(tok_pos_input)
    dep_results = dep_model.predict(tok_pos_inputs, task='dep')
    for result in dep_results:
        print(result)
    print('#' * 100)
    print('\n')

    # 加载SDP模型
    print('加载SDP模型...')
    config = json.load(open('pytorch_models/sdp/semeval16-news-biaffine_20191231_235407/config.json'))
    vocabs = json.load(open('pytorch_models/sdp/semeval16-news-biaffine_20191231_235407/vocabs.json'))
    sdp_model = BiaffineModel(config, vocabs)
    torch_init_model(sdp_model, 'pytorch_models/sdp/semeval16-news-biaffine_20191231_235407/model.pth')
    sdp_model.eval()
    tok_pos_inputs = []
    for tokens, poses in zip(cws_results, pos_results):
        tok_pos_input = []
        for tok, pos in zip(tokens, poses):
            tok_pos_input.append((tok, pos))
        tok_pos_inputs.append(tok_pos_input)
    sdp_results = sdp_model.predict(tok_pos_inputs, task='sdp')
    for result in sdp_results:
        print(result)

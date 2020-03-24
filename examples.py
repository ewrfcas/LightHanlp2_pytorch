from light_hanlp.models.conv_tagger import ConvTagger
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

    # 加载分词模型
    config = json.load(open('pytorch_models/cws/pku98_6m_conv_ngram_20200110_134736/config.json'))
    vocabs = json.load(open('pytorch_models/cws/pku98_6m_conv_ngram_20200110_134736/vocabs.json'))
    cws_model = ConvTagger(config, vocabs)
    torch_init_model(cws_model, 'pytorch_models/cws/pku98_6m_conv_ngram_20200110_134736/model.pth')
    cws_model.eval()

    cws_results = cws_model.predict(inputs)
    for result in cws_results:
        print(result)







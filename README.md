# LightHanlp2 pytorch
基于pytorch的轻量级hanlp2工具，支持中文分词，词性分类，实体抽取，句法分析，语义分析

感谢原项目作者的贡献https://github.com/hankcs/HanLP

## 版本依赖
pytorch >= 1.2.0

## 注意
本项目指在不依赖于tensorflow2.0轻便地调用hanlp2的模型，方便初心者理解各个工具的基本作用机理。~~另一个理由是个人使用hanlp2的时候存在内存溢出的问题，所以想用自己熟悉的结构来调用。~~ 并不提供训练等复杂功能(没有优化器配置，模型中没有配置dropout层)，完整功能请使用原hanlp2(https://github.com/hankcs/HanLP)。

## 模型下载地址


## 授人以鱼不如授人以渔
可以参考light_hanlp/utils/convert_keras_to_pytorch.py，将keras转化为pytorch模型。

```python
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

['H', 'an', 'L', 'P', '是', '一', '系列', '模型', '与', '算法', '组成', '的', '自然', '语言', '处理', '工具包', '，', '目标', '是', '普及', '自然', '语言', '处理', '在', '生产', '环境', '中', '的', '应用', '。']
['H', 'an', 'L', 'P', '具备', '功能', '完善', '、', '性能', '高效', '、', '架构', '清晰', '、', '语料', '时', '新', '、', '可自', '定义', '的', '特点', '。']
['内部', '算法', '经过', '工业界', '和', '学术界', '考验', '，', '配套', '书籍', '《', '自然', '语言', '处理', '入门', '》', '已经', '出版', '。']
['上海', '华安', '工业', '（', '集团', '）', '公司', '董事长', '谭旭光', '和', '秘书', '张晚霞', '来到', '美国', '纽约', '现代', '艺术', '博物馆', '参观', '。']
['萨哈夫', '说', '，', '伊拉克', '将', '同', '联合国', '销毁', '伊拉克', '大', '规模', '杀伤性', '武器', '特别', '委员会', '继续', '保持', '合作', '。']
['H', 'an', 'L', 'P', '支援', '臺', '灣', '正', '體', '、', '香港', '繁', '體', '，', '具有', '新', '詞', '辨', '識', '能力', '的', '中文', '斷', '詞', '系', '統']
```



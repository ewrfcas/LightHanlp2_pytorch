# LightHanlp2 pytorch
基于pytorch的轻量级hanlp2工具，支持中文分词，词性分类，实体抽取，句法分析，语义分析

感谢原项目作者的贡献https://github.com/hankcs/HanLP

## 版本依赖
pytorch >= 1.2.0

## 注意
本项目指在不依赖于tensorflow2.0轻便地调用hanlp2的模型，方便初心者理解各个工具的基本作用机理。~~另一个理由是个人使用hanlp2的时候存在内存溢出的问题，所以想用自己熟悉的结构来调用。~~ 并不提供训练等复杂功能(没有优化器配置，模型中没有配置dropout层)，完整功能请使用原hanlp2(https://github.com/hankcs/HanLP)。

## 模型下载地址
需要更多模型可以留言

## 授人以鱼不如授人以渔
可以参考fast_hanlp/utils/convert_keras_to_pytorch.py，将keras转化为pytorch模型。



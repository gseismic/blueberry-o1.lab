# 211_naive_zhdict_basicinfo

## 思考
偏好学习时一种或关系，而不是与关系。
对比学习时一种与关系，而不是或关系。
如果只用偏好学习: 能学到多解性，没有互斥，对比学习可以互斥，更有可能学到真正的逻辑

## Note
（1）发现直接在basic_full.txt上pretrain效果不好，于是尝试在basic_full_qa.txt上finetune。
（2）发现finetune后完全没有泛化能力，计划使用对比学习的方式来提升泛化能力。
    预计通过对比学习，可以提升模型在未见过的数据上的表现。计划使用DPO。
    - 发现错误的数据存在时，会导致错误的答案很难被学习（可能比正确答案更难学习），LLM具有识别错误答案的能力。
    - 把问题作为pretrain数据，导致答案的泛化能力很差，是这个原因？ 且没有添加ref_model约束。
（3）

## 直接finetune没有泛化能力的例子
```
**message["content"]='汉字`噙`的部首是`心部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`茛`的部首是`心部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`鲇`的部首是`欠部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`麂`的部首是`尸部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`鲇`的部首是`欠部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`麂`的部首是`尸部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`噙`的部首是`心部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`觐`的部首是`欠部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`痫`的部首是`心部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`茛`的部首是`心部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`噙`的部首是`心部`' != item["answer"]='汉字`歆`的部首是`欠部`'
```
持续继续finetune,可能原因: 问题部分的pretrain被过度优化
```
**message["content"]='汉字`觐`堇`的部首是`见部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`闩`的部首是`鱼部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`麂`麂`的部首是`鹿部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`闩`的部首是`门部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`您`您`的部首是`欠部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`麂`麂`的部首是`鹿部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`闩`的部首是`门部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`擘`擘`的部首是`手`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`闩`的部首是`鱼部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`闩`的部首是`门部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`序`序`的部首是`广部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`闩`的部首是`门部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`您`您`的部首是`欠部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`闩`的部首是`鱼部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`葩`葩`的部首是`虫部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`葩`葩`的部首是`氵部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`葩`葩`的部首是`虫部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`序`序`的部首是`广部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`觐`觐`的部首是`见部`' != item["answer"]='汉字`歆`的部首是`欠部`'
```
再继续finetune，loss从0.05降到0.0495左右
```

**message["content"]='汉字`琅`的部首是`王部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`闩`的部首是`门部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`阗`的部首是`疒部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`闩`的部首是`门部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`闩`的部首是`疒部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`琅`的部首是`王部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`阗`的部首是`疒部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`赫`的部首是`赤部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`阗`的部首是`门部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`闩`的部首是`门部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`揞`的部首是`扌部`' != item["answer"]='汉字`歆`的部首是`欠部`'
**message["content"]='汉字`痫`的部首是`疒部`' != item["answer"]='汉字`歆`的部首是`欠部`'
```

# 211_naive_zhdict_basicinfo
（1）发现直接在basic_full.txt上pretrain效果不好，于是尝试在basic_full_qa.txt上finetune。
（2）发现finetune后完全没有泛化能力，计划使用对比学习的方式来提升泛化能力。
    预计通过对比学习，可以提升模型在未见过的数据上的表现。计划使用DPO。
（3）


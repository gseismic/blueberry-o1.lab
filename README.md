# blueberry-o1.lab
code-NOT-ready, drafts
【未完成】

## 项目目标
openai发布o1模型之后，自己也有一些复现o1的思路，决定尝试复现。
因为算力和数据局限，虽然第一步就遭遇挫折，我还是决定把自己的思考写在这里，供大家批评和讨论。

传统LLM训练主要分两步：【预训练】和【微调】（含RLHF,DPO等）。

## 我个人对LM和AGI的思考:
- **预训练** + **微调** 好虽然目前效果很好，但并不符合人类的学习方法。预训练是一次性ALL-IN-ONCE投喂人类的所有知识，可能数T或数十T，而人类和动物都是**渐进式学习**。一次性训练一个超大数据集导致模型极度膨胀，对于简单闭环任务这是一个极大的算力和智慧的浪费。
而`渐进式学习`允许只学习`自己想知道/完成某项任务需要知道的知识`和`督导员指定的知识`，也可以逐步学习，这对`学习型机器人`可能十分重要。

- 为什么需要**超大规模**的预训练，是**神经网络**自身的特点或局限性造成的，基于两个观察和思考：
    - 以MLP为例，沿着【等值面】方向，只有**一阶精度**（高阶精度，计算上讲，可能性价比不高（lowrank效果带试），待验证），
      在划分边界的时候自由度过大，只有数据足够**稠密**，才能把足够精度的正确的**等值面**做出来
      这个等值面意味着不同概念的区分。（这可能也是很多传统训练都需要做《数据增强》的原因）
    - 借用SVM最大间距思想，如果数据量不够大，**对比学习**有可能能极大改善学习性能
      这和人类的学习方式类似，要知道什么是《猫🐱》，我们要知道《什么不是猫》，对比发现后，就能更容易学习到知识和**真理**

- 语言模型LM 能否实现通用智能AGI？虽然网上有一些其他言论，比如lecun的唱衰，我个人坚信**语言模型是AGI其中一种实现形式**。作为超级智能体人，我们是如何学习的呢？视觉——电磁波传入眼睛转为`电信号`，听觉——声波传入耳朵转为`电信号`，归一化的电信号是大脑流动的《语言》，我们只要确保输入和输出的一致性，外加神经网络的`万能表征`能力（特别是类transformer模型的高效表征能力），无论是语音、文字，还是视觉，都能够允许用统一的LM框架处理（各部分的Embeding除外）。（个人怀疑，多模态数据叠加时，类脑spiking可能是更高效的实现方法）

- transformer的本质：个人的理解，transformer的本质是符号计算，它高效的实现了模型内部多级IF-ELSE（MLP等也在内部通过relu实现IF-ELSE，但非常低效，每个if-else需要1层网络）。transformer的核心是attention，功能是讲，attention的是实现`基于内容索引`的一种有效机制，比如：本句的第一个字是__。就是索引和内容杂揉在一起，而不是像编程语言那样a[0], a[1], a[2]...。attention后的MLP实现的是二阶markov预测的logits [TODO: 增加参考链接，谋篇国外博客有讲解二阶markov https://e2eml.school/transformers.html#one_hot
]。

**总结**：以上3点思考后，得出自己个人化的结论：`渐进式LM`有助于实现不同级别的通用智能，训练出`蚂蚁智能`，`鹦鹉智能`，`猫狗智能`，`人类智能`，`超级智能`。（这里不叫Large LM是因为：渐进式LM模型可能并不大，甚至非常小，小到几百个神经元，数万参数）

## 渐进式智能如何实现？
### 方法1：通过最小预训练实现`渐进智能`
个人的LLM训练思路如下:
（1）学习【概念】：训练【基础语言模型】，此模型，我们需要的数据量只是一个比较完备的【最小字典】，效果上来说，我们要求字典是这样的：`我们学习完这个字典后，模型具有了学习新知识的能力`。这个字典可能类似于`新华字典`，只包含对某个单词的必要解释，包含基本的例句。
（2）学习【逻辑和知识】提供学前、小学、初中、高中、大学等渐进式数据，类似于强化学习领域的`课程学习`

（这个是大尺度的”基于过程“的训练）

注意：
    每次给一套《教材》学习，都要给出对应的《测试》，如果测试没有通过，需要找到对应问题。如果是旧知识忘了，要重新提供对应《教材》。
    这样，我们就清楚的知道，我们的模型在学习了什么资料后学会了什么，还有，忘记了什么。

### 方法2: 纯`渐进式`学习
 【方法1】的困难在于，能否找到这样的【最小字典】（我用blueberry训练失败的原因很可能就在于此，我个人抓取并使用的dict.cn字典过于简单，特别是缺少例句，导致训练失败，单片3060又难以支撑进行更大数据的训练）。这并不代表【方法1】不可行。
 直接做【方法1】的0-预训练的增量课程训练，核心工作在于构建【增量】数据。

## O1 复现方法
核心思路是：【宏观渐进学习】+ 【微观渐进学习】
    【宏观】（大尺度）渐进式学习（先学123数数，再学加减乘除，再微积分）
    【微观】语句级别渐进学习，比如 step-by-step论文
### 构建数据
   - 优先构建逻辑推理类数据，比如代码、数学、逻辑学
        - 代码： 
            - **编写代码解释器**，手动截获程序运行每一步后各变量的最新状态并以存储，得到超级详细的程序运行流程和结果
            - 在docker上运行各类程序，并向程序随机插入错误（类似于CriticGPT）
            - 模拟运代码截获可视化图，语言模型和多模态视觉模型可同时训练
        - 数学:
            - 生成推理数据 train step-by-step论文有介绍
        - 逻辑学:
            - 让模型学会演绎法和归纳法，各种逻辑推断真值表
    - 构建【想象力左脑】和【批判性右脑】
        - 【想像力模型】用于生成更丰富的数据，【扩大】想像力模型的生成范围
        - 【批判力模型】用于否定【想像力模型】的生成结果，用于【缩减】【想像力模型】的生成范围
        - 这两个模型可以考虑合二为一，有两种激活状态，可以手动激活，也可以自适应激活

### token级渐进强化学习
token级强化学习的例子：

一个完整【一步】是：
 【proposer提议step】【critic质证必要性】【judge判断改步骤是否有必要进行，过程是否终止】
具体训练，也使用强化学习训练。
<system/>...</system><user/>...</user>
    <不可见的think>/><step_proposal/>..</step_proposal><step_critic/>..</step_critic><judge/>..</judge>..<terminated/truncated></</think>
    <可见的output/></output>

因算力、数据和个人精力问题，这部分一直没有实施。


## 代码部分参考资料
[naive代码来自chatgpt_3k.py] chatgpt_3k.py
[llama3](https://github.com/meta/llama)
[nanogpt](https://github.com/karpathy/nanoGPT)
[minimind](https://github.com/jingyaogong/minimind/blob/master/model/model.py)
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt2.py)

## 相关论文
（来自某github仓库, TODO: 添加链接）

[Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://arxiv.org/pdf/2405.18392)

《从零实现强化学习、RLHF、AlphaZero》-1：基于价值的强化学习1-理论基础、q learning、sarsa、dqn
https://zhuanlan.zhihu.com/p/673543350

Awesome-LLM-Strawberry
https://github.com/hijkzzz/Awesome-LLM-Strawberry?tab=readme-ov-file

Transformers from Scratch物理意义解释
https://e2eml.school/transformers.html#one_hot

How ChatGPT is fine-tuned using Reinforcement Learning
https://dida.do/blog/chatgpt-reinforcement-learning

How ChatGPT actually works
https://www.assemblyai.com/blog/how-chatgpt-actually-works/

OpenRLHF
https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/dpo_trainer.py

AlphaZero-Like Tree-Search can Guide
Large Language Model Decoding and Training
https://arxiv.org/pdf/2309.17179

Training Verifiers to Solve Math Word Problems
https://arxiv.org/pdf/2110.14168

Generative Language Modeling for Automated Theorem Proving
https://arxiv.org/abs/2009.03393

LLM Critics Help Catch LLM Bugs
https://arxiv.org/pdf/2407.00215

Self-critiquing models for assisting human evaluators
https://arxiv.org/pdf/2206.05802

Scalable Online Planning
via Reinforcement Learning Fine-Tuning
https://arxiv.org/pdf/2109.15316

Q*: Improving Multi-step Reasoning for LLMs with
Deliberative Planning https://arxiv.org/pdf/2406.14283

Let’s Verify Step by Step
https://arxiv.org/pdf/2305.20050

Don’t throw away your value model!
Generating more preferable text with Value-Guided Monte-Carlo
Tree Search decoding
https://arxiv.org/pdf/2309.15028

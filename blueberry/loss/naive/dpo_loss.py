
import torch
import torch.nn.functional as F


def dpo_loss(model, reference_model, prompts, chosen_responses, rejected_responses, beta):
    """
    计算 DPO 损失函数

    参数:
    - model: 目标语言模型（pi_theta）
    - reference_model: 参考语言模型（pi_ref）
    - prompts: 输入提示 x
    - chosen_responses: 人类偏好的回答 y_w
    - rejected_responses: 人类不偏好的回答 y_l
    - beta: 控制偏离程度的超参数

    返回:
    - 损失值
    """
    # 获取目标模型对 chosen_responses 和 rejected_responses 的 logits
    chosen_logits = model(prompts, labels=chosen_responses)
    rejected_logits = model(prompts, labels=rejected_responses)

    # 获取参考模型对 chosen_responses 和 rejected_responses 的 logits
    ref_chosen_logits = reference_model(prompts, labels=chosen_responses)
    ref_rejected_logits = reference_model(prompts, labels=rejected_responses)

    # 计算 log 概率比率
    log_ratio_chosen = F.log_softmax(chosen_logits, dim=-1) - F.log_softmax(ref_chosen_logits, dim=-1)
    log_ratio_rejected = F.log_softmax(rejected_logits, dim=-1) - F.log_softmax(ref_rejected_logits, dim=-1)

    # 计算 DPO 损失
    preference_score = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -torch.log(torch.sigmoid(preference_score)).mean()

    return loss
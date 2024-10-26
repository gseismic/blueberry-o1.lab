# pretrain
pretrain_data_file = '../../datasets/zhdict/basic_full.txt'
pretrain_text_sep = '|||\n\n'

# final model file
final_model_file = 'models/final_model.pth'
checkpoint_freq = 10

# ** 字典太大，语料太少的原因
seq_len = 100

gpt_config = {
    "num_layers": 6, # 12, #
    "embed_dim": 12*10*5, # 768, memory影响不大？
    "num_heads": 6, # 12,
    "ff_dim": 512,
    "seq_len": seq_len, # 256, # 768, # **计算量n**2
    "dropout": 0.1
}

# NOTE: 最后可考虑使用最后k步的平均模型作为最终输出
# 动量具有batch_size效果，batch_size不必太大
pretrain_initial_model = None
# XXX continue-training
# pretrain_initial_model = final_model_file
# pretrain_initial_model = 'models/checkpoint/chkpt_40.pth'
# 继续训练，如果没有warmup，动量还没有积累，会出现陡增的情况
# warmup数和动量大小相关，adam动量越大,warmup越大
pretrain_config = {
    'device': 'cuda',
    'seed': 36,
    'lr': 1e-4, # 理论上，batch_size大，lr更大
    'warmup_epochs': 1, # 10
    'grad_clip': 1, # 这个数值理论上也应该 decrease
    'batch_size': 32, # memory: linear，computation time not influenced
    'max_epochs': 2, # 100
    'target_loss_ratio': 0.001,
    'target_loss': 0.001,
    'verbose_freq': 1,
    'batch_verbose_freq': 50,
}

eval_model_file = 'models/checkpoint/chkpt_180.pth'
eval_config = {
    "start_texts": [
        "`镓`",
        "汉字`镓`",
        "汉字`中`",
        "中",
        # "汉字`汉`",
        # "笔画数",
        # "国",
        # "汉字`人`",
        # "人",
        # "汉字`民`",
        # "汉字`我`",
    ]
}

finetune_initial_model = 'models/final_model.pth'
finetune_ref_model = 'models/final_model.pth' # XXX, TODO 没有加ref_model约束
finetune_final_model_file = 'models/final_model_finetune.pth'
finetune_checkpoint_dir = 'models/checkpoint_finetune'
finetune_data_file = '../../datasets/zhdict/basic_full_qa.txt'
finetune_text_sep = '|||\n\n'
finetune_config = {
    'seed': 36,
    'batch_size': 32,
    'max_epochs': 30,
    'lr': 1e-4,
    'warmup_epochs': 5,
    'grad_clip': 1,
    'target_loss_ratio': 0.001,
    'target_loss': 0.001,
    'verbose_freq': 10,
    'batch_verbose_freq': 10,
    'checkpoint_freq': 5,
}

# finetune_eval_model_file = 'models/final_model_finetune.pth'
finetune_eval_model_file = 'models/checkpoint_finetune/chkpt_30.pth'
finetune_eval_data_file = '../../datasets/zhdict/basic_full_qa_eval.txt'
# finetune_eval_data_file = '../../datasets/zhdict/basic_full_qa.txt'
finetune_eval_text_sep = '|||\n\n'
eval_finetune_config = {
    'max_generate_len': 50,
}

dpo_data_file = '../../datasets/zhdict/basic_full_dif_qa.txt'
dpo_text_sep = '|||\n\n'

# dpo_initial_model = 'models/final_model_finetune.pth'
# dpo_ref_model = 'models/final_model_finetune.pth'
# 直接使用pretrain模型
# dpo_initial_model = 'models/final_model.pth'
# dpo_ref_model = 'models/final_model.pth'
dpo_initial_model = 'models/checkpoint_finetune/chkpt_15.pth'
dpo_ref_model = 'models/checkpoint_finetune/chkpt_15.pth'

dpo_final_model_file = 'models/final_model_dpo.pth'
dpo_checkpoint_dir = 'models/checkpoint_dpo'
dpo_config = {
    'seed': 36,
    'beta': 0.1,
    'batch_size': 32,
    'max_epochs': 200,
    'lr': 1e-5, #  when finetunning: make smaller 
    'warmup_epochs': 50,
    'grad_clip': 1,
    'target_loss_ratio': 0.001,
    'target_loss': 0.001,
    'verbose_freq': 10,
    'batch_verbose_freq': 10,
    'checkpoint_freq': 10,
}

# dpo_eval_data_file = '../../datasets/zhdict/basic_full_dpo_eval.txt'
dpo_eval_data_file = '../../datasets/zhdict/basic_full_qa_eval.txt'
dpo_eval_text_sep = '|||\n\n'
dpo_eval_config = {
    'max_generate_len': 50,
}

# pretrain
pretrain_data_file = '../../datasets/zhdict/basic_full.txt'
pretrain_text_sep = '|||\n\n'

# final model file
final_model_file = 'models/final_model.pth'
checkpoint_freq = 10

# ** 字典太大，语料太少的原因
seq_len = 40

gpt_config = {
    "num_layers": 6, # 12, #
    "embed_dim": 12*10*5, # 768, memory影响不大？
    "num_heads": 6, # 12,
    "ff_dim": 512,
    # "seq_len": 32, # 256, # 768, # **计算量n**2
    # "dropout": 0.1
}

# NOTE: 最后可考虑使用最后k步的平均模型作为最终输出
# 动量具有batch_size效果，batch_size不必太大
# pretrain_initial_model = None
# XXX continue-training
# pretrain_initial_model = final_model_file
pretrain_initial_model = 'models/checkpoint/chkpt_40.pth'
# 继续训练，如果没有warmup，动量还没有积累，会出现陡增的情况
# warmup数和动量大小相关，adam动量越大,warmup越大
pretrain_config = {
    'device': 'cuda',
    'seed': 36,
    'lr': 1e-4, # 理论上，batch_size大，lr更大
    'warmup_epochs': 10,
    'grad_clip': 1, # 这个数值理论上也应该 decrease
    'batch_size': 32, # memory: linear，computation time not influenced
    'max_epochs': 500,
    'target_loss_ratio': 0.001,
    'target_loss': 0.001,
    'verbose_freq': 1,
    'batch_verbose_freq': 10,
}

eval_model_file = 'models/checkpoint/chkpt_130.pth'
eval_config = {
    "start_texts": [
        "汉字`中`",
        "中",
        #
    ]
}


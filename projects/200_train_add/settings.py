
# pretrain
pretrain_data_file = 'dataset/main.txt'
pretrain_text_sep = '|||\n\n'

# final model file
final_model_file = 'models/final_model.pth'

# GP conofig
# TODO: 这些参数的影响分别是什么
# TODO: loss mask
gpt_config = {
    "num_layers": 2, # 12,
    "embed_dim": 12*20, # 768,
    "num_heads": 4, # 12,
    "ff_dim": 512,
    "seq_len": 20, # **
    "dropout": 0.1
}

pretrain_config = {
    'device': 'cuda',
    'seed': 36,
    'batch_size': 128,
    'max_epochs': 3_000,
    'target_loss_ratio': 0.001,
    'target_loss': 0.001,
    'verbose_freq': 10
}


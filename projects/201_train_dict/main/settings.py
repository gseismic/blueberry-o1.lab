
# pretrain
pretrain_data_file = 'dataset/main.txt'
pretrain_text_sep = '|||\n\n'

# final model file
final_model_file = 'models/final_model.pth'

# GP conofig
# TODO: 这些参数的影响分别是什么
# TODO: loss mask
gpt_config = {
    "num_layers": 12,
    "embed_dim": 768,
    "num_heads": 12,
    "ff_dim": 512,
    "seq_len": 50,
    "dropout": 0.1
}

pretrain_config = {
    'seed': 36,
    'batch_size' = 160 # 10
max_epochs = 6_00
target_loss_ratio = 0.001
target_loss = 0.001
}

# pretrain_config = {
# }

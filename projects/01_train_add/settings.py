
pretrain_data_file = 'dataset/main.txt'
text_sep = '|||\n\n'

final_model_file = 'models/final_model.pth'

gpt_config = {
    "num_layers": 2,
    "embed_dim": 128,
    "num_heads": 4,
    "ff_dim": 512,
    "seq_len": 20,
    "dropout": 0.1
}

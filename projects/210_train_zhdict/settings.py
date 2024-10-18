import socket
# pretrain
# pretrain_data_list_dir = '/Users/mac/turing/llm_dataset/dictcn/hanyu/naive/list'
_host = socket.gethostname()
if _host == 'quant':
    pretrain_data_file = '/home/lsl/turing/llm_dataset/dictcn/hanyu/naive/full.txt'
else:
    pretrain_data_file = '/Users/mac/turing/llm_dataset/dictcn/hanyu/naive/full.txt'

pretrain_text_sep = '|||\n\n'

# final model file
final_model_file = 'models/final_model.pth'
checkpoint_freq = 10

# GP conofig
# TODO: 这些参数的影响分别是什么
# TODO: loss mask
# gpt_config = {
#     "num_layers": 4, # 12, # 
#     "embed_dim": 12*10, # 768, memory影响不大？
#     "num_heads": 12, # 12,
#     "ff_dim": 512,
#     "seq_len": 256, # 768, # **计算量n**2
#     "dropout": 0.1
# }

gpt_config = {
    "num_layers": 4, # 12, # 
    "embed_dim": 12, # 768, memory影响不大？
    "num_heads": 6, # 12,
    "ff_dim": 512,
    "seq_len": 32, # 256, # 768, # **计算量n**2
    "dropout": 0.1
}

pretrain_initial_model = None
# XXX continue-training
# pretrain_initial_model = final_model_file
pretrain_config = {
    'device': 'cuda',
    'seed': 36,
    'lr': 0.00025, # 理论上，batch_size大，lr更大
    'batch_size': 32, # memory: linear，computation time not influenced
    'max_epochs': 1000,
    'target_loss_ratio': 0.001,
    'target_loss': 0.001,
    'verbose_freq': 1
}

eval_config = {
    "start_texts": [
        "一",
        "`一`的意思",
        # "`多`",
    ]
}


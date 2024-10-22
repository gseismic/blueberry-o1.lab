from torch.utils.data import DataLoader
from blueberry.data.pretrain import MemDataset
from blueberry.tokenizer.naive import Tokenizer

pretrain_data_file = './data.txt'

pretrain_text_sep = '|||\n\n'
seq_len = 10
batch_size = 2
# ** init dataset **
tokenizer = Tokenizer.from_files([pretrain_data_file], 
                                pretrain_text_sep=pretrain_text_sep)
dataset = MemDataset.from_file(pretrain_data_file, 
                               tokenizer=tokenizer,
                               seq_len=seq_len, 
                               pretrain_text_sep=pretrain_text_sep)
# print(dataset[0])
for input_seq, target_seq in dataset:
    print('='*20)
    print(f'{input_seq=}')
    print(f'{target_seq=}')
    print(type(input_seq))
    print(f'{tokenizer.decode(input_seq.numpy().tolist())=}')
    print(f'{tokenizer.decode(target_seq.numpy().tolist())=}')
    

print(f'{len(dataset)=}')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 

for input_seq, target_seq in data_loader:
    print(f'{input_seq.shape=}, {target_seq.shape=}')

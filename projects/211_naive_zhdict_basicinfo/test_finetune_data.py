from pathlib import Path
import settings
from blueberry.tokenizer.naive import Tokenizer
from blueberry.data.finetune.naive.mem import FinetuneDataset

def main():
    filename = Path(settings.finetune_data_file)
    if not filename.exists(): 
        raise FileNotFoundError(f"Finetune data file not found: {filename}")
    
    tokenizer = Tokenizer.from_files([settings.pretrain_data_file], 
                                    pretrain_text_sep=settings.pretrain_text_sep)
    dataset = FinetuneDataset.from_file(settings.finetune_data_file, 
                                       tokenizer=tokenizer,
                                       max_seq_len=settings.seq_len, 
                                       data_sep=settings.finetune_text_sep)
    print(f'len(dataset): {len(dataset)}')
    # print(f'dataset[0]: {dataset[0]}')
    # for item in dataset:
    #     print(item)


if __name__ == '__main__':
    main()

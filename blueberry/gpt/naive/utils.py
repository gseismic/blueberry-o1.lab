from ..config import torch


def pad_sequences(sequences, max_len, pad_index):
    '''
    # 使用示例
    input_ids = [[1, 2, 3], [4, 5], [6]]
    max_len = 5
    pad_index = 0
    padded_sequences = pad_sequences(input_ids, max_len, pad_index)
    print(padded_sequences)

    out:
        tensor([[1, 2, 3, 0, 0],
                [4, 5, 0, 0, 0],
                [6, 0, 0, 0, 0]])
    '''
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            padded_seq = seq[:max_len]  # 截断到最大长度
        else:
            padded_seq = seq + [pad_index] * (max_len - len(seq))  # 填充到最大长度
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences)

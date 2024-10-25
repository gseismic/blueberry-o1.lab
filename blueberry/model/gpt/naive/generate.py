from ....config import torch, F


def generate_sequence(model, start_tokens, max_gen_len, vocab_size, 
                      max_seq_len=None,
                      device='cuda',
                      temperature=1.0, top_k=None, top_p=None, callback=None,
                      stop_at_eos=False, eos_token_id=None):
    """
    生成文本序列。

    Parameters:
    - model: GPT模型
    - start_tokens: 起始token的索引
    - max_gen_len: 生成序列的最大长度
    - vocab_size: 词汇表大小
    - device: 设备（如 'cuda' 或 'cpu'）
    - temperature: 温度调整参数 (默认为1.0)
    - top_k: Top-k采样的k值，如果为None则不使用Top-k采样
    - top_p: Top-p采样的p值，如果为None则不使用Top-p采样

    Returns:
    - 生成的token序列

    使用示例
    ```
        embed_dim = 512
        num_heads = 8
        num_layers = 6
        ff_dim = 2048
        vocab_size = 10000
        dropout = 0.1
        max_gen_len = 20
        start_tokens = 0  # 起始token索引

        # 假设您已经定义并加载了 GPT 模型
        gpt = GPT(num_layers, embed_dim, num_heads, ff_dim, vocab_size, dropout)

        # 温度调整生成
        generated_sequence = generate_sequence(gpt, start_tokens, max_gen_len, vocab_size, device='cpu', temperature=0.7)
        print("Generated sequence with temperature:", generated_sequence)

        # Top-k 采样生成
        generated_sequence = generate_sequence(gpt, start_tokens, max_gen_len, vocab_size, device='cpu', top_k=50)
        print("Generated sequence with top-k sampling:", generated_sequence)

        # Top-p 采样生成
        generated_sequence = generate_sequence(gpt, start_tokens, max_gen_len, vocab_size, device='cpu', top_p=0.9)
        print("Generated sequence with top-p sampling:", generated_sequence)
    """
    assert isinstance(device, torch.device) or device in ['cpu', 'cuda']
    assert not (top_k is True and top_p is True)
    assert top_p is None or 0 < top_p < 1
    assert top_k is None or 0 < top_k <= max_gen_len
    device = torch.device(device) if isinstance(device, str) else device
    # assert isinstance(start_tokens, list)
    model.eval()
    if isinstance(start_tokens, (list,tuple)):
        start_tokens = list(start_tokens)
    else:
        start_tokens = [start_tokens]
    
    generated = start_tokens[:]
    if max_seq_len is None:
        input_seq = torch.tensor(generated[:], device=device).unsqueeze(0)  # [1, 1]
    else:
        input_seq = torch.tensor(generated[-max_seq_len:], device=device).unsqueeze(0)  # [1, 1]

    with torch.no_grad():
        for i in range(max_gen_len - 1):
            if max_seq_len is None:
                input_seq = torch.tensor(generated[:], device=device).unsqueeze(0)  # [1, 1]
            else:
                input_seq = torch.tensor(generated[-max_seq_len:], device=device).unsqueeze(0)  # [1, 1]

            output = model(input_seq)  # [1, seq_len, vocab_size]
            logits = output[0, -1, :]  # [vocab_size]
            
            # 应用温度调整
            if temperature != 1.0:
                logits /= temperature

            probs = F.softmax(logits, dim=-1)
            #print(f'{probs=}')
            #print(f'{sorted(probs, reverse=True)=}')
            #print(f'{torch.sum(probs)=}')
            #print(f'{vocab_size=}')
            
            if top_k is not None:
                # Top-k采样
                # logits: [vocab_size]
                # top_k_values: [50]
                # print(f'{logits=}, {logits.shape=}')
                # print(f'{top_k=}')
                top_k_values, top_k_indices = logits.topk(top_k)
                # print(f'{top_k_values=}')
                top_k_probs = F.softmax(top_k_values, dim=-1)
                # print(f'{top_k_probs=}')
                # 采样1个
                next_token = top_k_indices[torch.multinomial(top_k_probs, 1).item()]
            elif top_p is not None:
                # Top-p采样
                # print(f'{probs.shape=}')
                # [vocab_size]
                # 概率累积求和，因为是一次只预测一个token
                # https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p#2-pick-from-amongst-the-top-tokens-top-k
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # print(f'{cumulative_probs=}, {top_p=}')
                sorted_indices_to_keep = cumulative_probs <= top_p
                # print(f'{sorted_indices_to_keep=}, {sorted_indices_to_keep.shape=}')
                top_p_indices = sorted_indices[sorted_indices_to_keep]
                top_p_probs = sorted_probs[sorted_indices_to_keep]
                # Ensure at least one token is kept
                # 如果为空，会导致RuntimeError
                if len(top_p_probs) > 0:
                    next_token = top_p_indices[torch.multinomial(top_p_probs, 1).item()]
                    # print(f'{next_token=}')
                else:
                    # top_p_probs: [0.7916, 0.9079, 0.9552..], top_p = 0.6
                    # 代表第一个的概率超过0.6了
                    # print(f'**Warning: size(top_p_probs) == 0, select the first token')
                    next_token = sorted_indices[0]
            else:
                # 基于温度调整的采样
                next_token = torch.multinomial(probs, 1)

            generated.append(next_token.item())
            if callback is not None:
                should_stop = callback(next_token.item())
                if should_stop is True:
                    break
            input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=device)], dim=1)
            if stop_at_eos and eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return generated

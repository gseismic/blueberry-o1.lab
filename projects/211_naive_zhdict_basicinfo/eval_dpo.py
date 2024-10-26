from blueberry.model.gpt.naive import GPT
from blueberry.tokenizer.naive import Tokenizer, ChatFormat, Dialog, Message
from blueberry.tokenizer.naive.exception import ParseDialogError
from blueberry.utils.xz import ensure_xzfile_decompressed
import settings
import json

# ** decompress xz file **  
ensure_xzfile_decompressed(settings.pretrain_data_file)
tokenizer = Tokenizer.from_files([settings.pretrain_data_file],
                                pretrain_text_sep=settings.pretrain_text_sep)
# gpt = GPT.load(settings.final_model_file)
gpt = GPT.load(settings.dpo_final_model_file)
print(f'{settings.dpo_final_model_file=}')

# ** decompress xz file **  
ensure_xzfile_decompressed(settings.dpo_eval_data_file)
def get_data():
    with open(settings.dpo_eval_data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    return [json.loads(t.strip()) for t in text.split(settings.dpo_eval_text_sep)]

data = get_data()[:100]
# XXX: 只测试前100条数据
# data = [data[0]]*100
print(f'{len(data)=}')
# print(data)
correct, total = 0, 0
chat_format = ChatFormat(tokenizer)
for item in data:
    # print('='*100)
    # print(item)
    diag = [
        Message(role='系统', content=''), 
        Message(role='用户', content=item['question'])
    ]
    start_tokens, info = chat_format.encode_dialog_prompt(diag, assistant_role='助手')

    max_generate_len = settings.dpo_eval_config['max_generate_len']
    generated_sequence = gpt.generate(start_tokens, 
                                     max_gen_len=max_generate_len,
                                     temperature=0.5, top_k=10,
                                     stop_at_eos=True, eos_token_id=tokenizer.end_text_id)
    # print(f'{generated_sequence=}, {len(generated_sequence)=}')
    # print(f'{tokenizer.decode(generated_sequence, skip_all_special=False)=}')
    try:
        decoded_dialog = chat_format.decode_dialog(generated_sequence)
        # print(f'{decoded_dialog=}')
        for message in decoded_dialog:
            if message['role'] == '助手':
                # print(f'{message["content"]=}')
                # print(f'{item["answer"]=}')
                if message['content'] == item['answer']:
                    print(f'correct')
                    correct += 1
                else:
                    print(f'**{message["content"]=} != {item["answer"]=}')
                total += 1
    except ParseDialogError as e:
        print(f'{e=}')
print(f'{correct=}, {total=}')

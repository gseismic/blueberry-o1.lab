from pathlib import Path
import settings
from collections import Counter

def main():
    filename = Path(settings.pretrain_data_file)
    if not filename.exists():
        raise FileNotFoundError(f"Pretrain data file not found: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    texts = text.split(settings.pretrain_text_sep)
    print(f'{len(texts)} texts found.')
    
    token_counter = Counter()
    for text in texts:
        # print(text)
        token_counter.update(text)
    
    print(token_counter)
    print(f'{len(token_counter)} unique tokens found.')

if __name__ == '__main__':
    main()

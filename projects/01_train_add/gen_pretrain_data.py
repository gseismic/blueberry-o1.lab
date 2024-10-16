from pathlib import Path
import settings


def make_data():
    data = []
    for i in range(10):
        for j in range(10):
            z = i + j
            line1 = f'{z} = {i} + {j}'
            line2 = f'{i}+{j}={z}'
            data.append(line1)
            data.append(line2)
    return data


def main():
    filename = Path(settings.pretrain_data_file)
    filename.parent.mkdir(exist_ok=True)
    print('Generate pretrain data ...')
    with open(filename, 'w') as f:
        text = settings.pretrain_text_sep.join(make_data())
        f.write(text)
        
if __name__ == '__main__':
    main()

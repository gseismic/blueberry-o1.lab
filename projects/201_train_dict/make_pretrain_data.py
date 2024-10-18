import glob
import json
from pathlib import Path
from main import settings


root_path = '/Users/mac/turing/llm_dataset/dictcn/hanyu'
def read_file(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def make_data():
    filepaths = glob.glob(root_path+ '/*.json')
    print(filepaths)
    for path in filepaths:
        print(path)
        data = read_file(path)
        print(data)

def main():
    make_data()
    return

    filename = Path(settings.pretrain_data_file)
    filename.parent.mkdir(exist_ok=True)
    print('Generate pretrain data ...')
    with open(filename, 'w') as f:
        text = settings.pretrain_text_sep.join(make_data())
        f.write(text)
    print('Done!')
        
if __name__ == '__main__':
    main()

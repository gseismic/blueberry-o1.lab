#from pathlib import Path

#def load_data(filename):
#    print(filename)
#    with open(filename) as f:
#        lines = f.readlines()
#    return [line.strip() for line in lines]
#
#HAN3500 = load_data(Path.joinpath(Path(__file__).parent, 'txt/han3500.txt'))
#HAN7000 = load_data(Path.joinpath(Path(__file__).parent, 'txt/han7000.txt'))
from .han3500 import HAN3500
from .han7000 import HAN7000

__all__ = ['HAN3500', 'HAN7000']

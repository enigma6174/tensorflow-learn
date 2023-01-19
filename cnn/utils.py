import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path
from collections import Counter


def get_dir():
    '''
    Function to return the current working directory
    '''
    directory = Path.cwd()
    return directory

def tree(_dir, filetype):
    '''
    Function to display the expanded directory in tree view along with file count
    '''
    print(f'+ {_dir}')
    for path in sorted(_dir.rglob('')):
        depth = len(path.relative_to(_dir).parts)
        count = Counter(p.suffix for p in path.glob(filetype))
        spacer = '   ' * depth
        key = filetype[1:]
        print(f'{spacer}+ {path.name} ({count[key]} files)')
        
def generate_classes(_dir):
    '''
    Function to generate classes from the data directory
    '''
    classes = np.array(sorted([item.name for item in _dir.glob('*')]))
    return classes

def generate_plots(_dir, _cls):
    '''
    Function to generate a 3x3 plot of random images of a given class
    '''
    samples = random.sample([item for item in sorted((_dir/_cls).glob('*'))], 9)
    plt.figure(figsize=(16,16))
    for i in range(9):
        img = mpimg.imread(samples[i])
        plt.subplot(3,3,i+1)
        plt.imshow(img)
        plt.title(f'class: {_cls} shape: {img.shape}')
        plt.axis('off')
    plt.show()
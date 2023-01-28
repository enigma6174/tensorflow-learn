import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path
from collections import Counter
from tensorflow.io import read_file
from tensorflow.image import decode_image, resize

def get_dir(dataset):
    '''
    Return the current working directory
    '''
    directory = Path.cwd()/'data'/dataset
    return directory

def tree(_dir, filetype):
    '''
    Display the expanded directory in tree view along with file count
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
    Generate classes from the data directory
    '''
    classes = np.array(sorted([item.name for item in _dir.glob('*')]))
    return classes

def generate_plots(_dir, _cls):
    '''
    Generate a 3x3 plot of random images of a given class
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
    
def plot_history(history):
    '''
    Returns separate loss curves for training and validation metrics
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    epochs = range(len(history.history['loss']))
    
    plt.figure(figsize=(12,6))
    
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label='train')
    plt.plot(epochs, val_loss, label='validation')
    plt.title('LOSS_PLOT')
    plt.xlabel('epochs')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs, acc, label='train')
    plt.plot(epochs, val_acc, label='validation')
    plt.title('ACCURACY_PLOT')
    plt.xlabel('epochs')
    plt.legend()
    
    plt.show()

def clean_data(_dir):
    '''
    Removes image files that cannot be opened or have been corrupted
    '''
    try:
        try:
            for item in (_dir.glob('*')):
                folder_name = str(item).split('/')[-1]
                if folder_name == '.DS_Store':
                    print(f'[HIDDEN_DIR] DELETING UNWANTED DIRECTORY\n{item}')
                    item.unlink()
        except Exception as e:
            print(f'[ERR] ERROR IN DIR: {e}')

        else:
            for image in sorted(_dir.glob('*')):
                try:
                    img = read_file(str(image))
                    img = decode_image(img)

                    if img.ndim != 3:
                        print(f"[FILE_CORRUPT] {str(image).split('/')[-1]} DELETED")
                        image.unlink()

                except Exception as e:
                    print(f"[ERR] {str(image).split('/')[-1]}: {e} DELETED")
                    image.unlink()
    except Exception as e:
        print(f'[ERR] UNEXPECTED ERROR OCCURRED: {e}')
            
def process_image(file, img_shape=224):
    '''
    Process the image before loading into the model for prediction
    '''
    try:
        img = read_file(str(file))
        img = decode_image(img)
    except Exception as e:
        print(f"[ERR] file: {file} : {e}")
    else:
        img = resize(img, [img_shape, img_shape])
        img = img/255.
        
    return img
    
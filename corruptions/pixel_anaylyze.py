import matplotlib.pyplot as plt
import argparse
from PIL import Image
import numpy as np
import mmcv
from petrel_client.client import Client

client = Client('~/petreloss.conf')

def get_args():
    parser = argparse.ArgumentParser(description='Create RoboDepth Corruptions')
    # general configurations
    parser.add_argument('--image_list', type=str,
                        help="the file path to the image list.", default="./filename.txt")       
    parser.add_argument('--save_path', type=str,
                        help="the file path for saving histograms.", default="./test.png")
    parser.add_argument('--title', type=str,
                        help="the images title.", default="Higtogram of Pixel Values")
    return parser.parse_args()


def get_files(image_list):
    with open(image_list, 'r') as f:
        filenames = f.readlines()
        filenames = [file.strip() for file in filenames]
    return filenames


def get_histgram(args):
    filenames = get_files(args.image_list)
    histograms = []
    rand_filenames = np.random.choice(filenames, 1000)
    progress_bar = mmcv.utils.ProgressBar(len(rand_filenames))
    for i, file in enumerate(rand_filenames):
        # image = np.array(Image.open(file))
        img_bytes = client.get(file)
        image = mmcv.imfrombytes(img_bytes)
        histogram_, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))
        histograms.append(histogram_)
        progress_bar.update()
    histograms = np.array(histograms).mean(axis=0)
    plt.bar(bins[:-1], histograms, width=1, alpha=0.8)
    plt.xlabel('Pixel Value', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.grid(alpha=0.3)
    plt.ylim(0, 200000)
    plt.title(args.title, fontsize=18)
    plt.xticks(fontsize=15)  # Adjust the font size
    plt.yticks(fontsize=15)  # Adjust the font size
    plt.tight_layout()
    plt.savefig(args.save_path)
    exit()
        



if __name__=='__main__':
    args = get_args()
    get_histgram(args)
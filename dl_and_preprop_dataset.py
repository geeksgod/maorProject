#!/usr/bin/env python

__author__ = 'Erdene-Ochir Tuguldur'

import os
import sys
import csv
import time
import argparse
import fnmatch
import librosa
import pandas as pd

from hparams import HParams as hp
from zipfile import ZipFile
from audio import preprocess
from utils import download_file
from datasets.np_speech import NPSpeech

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", required=True, choices=['NPSpeech'], help='dataset name')
args = parser.parse_args()

args.dataset == 'NPSpeech':
dataset_name = 'NPSpeech-1.0'
datasets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
dataset_path = os.path.join(datasets_path, dataset_name)

   

# pre process
print("pre processing...")
np_speech = NPSpeech([])
preprocess(dataset_path, np_speech)

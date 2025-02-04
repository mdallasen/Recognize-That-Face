import argparse
import os
from functions.detect_label import detect_label
from functions.train_model import train_model
from functions.recognize import recognize

def main(args):

    if args.mode == "train": 
        train_model(args.dataset_path)


        
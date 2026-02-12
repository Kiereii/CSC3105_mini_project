"""
Created on Feb 6, 2017

@author: Klemen Bregar 
"""

import os
import pandas as pd
from numpy import vstack


def import_from_files():
    """
        Read .csv files and store data into an array
        format: |LOS|NLOS|data...|
    """
    # resolve dataset directory relative to this script file
    script_dir = os.path.dirname(__file__)
    rootdir = os.path.abspath(os.path.join(script_dir, '..', 'dataset'))

    output_arr = None
    for dirpath, dirnames, filenames in os.walk(rootdir):
        for file in filenames:
            # only process CSV files
            if not file.lower().endswith('.csv'):
                continue
            filename = os.path.join(dirpath, file)
            print('Reading:', filename)
            # read data from file
            df = pd.read_csv(filename, sep=',', header=0)
            input_data = df.values
            # append to array
            if output_arr is None:
                output_arr = input_data
            else:
                output_arr = vstack((output_arr, input_data))

    # return empty list when no files found to avoid IndexError
    if output_arr is None:
        return []

    return output_arr

if __name__ == '__main__':

    # import raw data from folder with dataset
    print("Importing dataset to numpy array")
    print("-------------------------------")
    data = import_from_files()
    print("-------------------------------")
    # print dimensions and data
    print("Number of samples in dataset: %d" % len(data))
    print("Length of one sample: %d" % len(data[0]))
    print("-------------------------------")
    print("Dataset:")
    print(data)
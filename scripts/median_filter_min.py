import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def median1d(arr, k):
    w = len(arr)
    idx = np.fromfunction(lambda i, j: i + j, (k, w), dtype=int) - k // 2
    idx[idx < 0] = 0
    idx[idx > w - 1] = w - 1
    return np.median(arr[idx], axis=0)

directory = '/home/takayanagi/data/diff/kyoto/1hour/'

save_dir = '/home/takayanagi/data/diff_clean/kyoto/1hour/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def process_file(filename):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        data = np.loadtxt(filepath, delimiter=',', skiprows=1, max_rows=5000)
        data[1:, 3] = median1d(data[1:, 3], 5)

        save_path = os.path.join(save_dir, filename)
        np.savetxt(save_path, data, delimiter=',', fmt='%.6f')

with ProcessPoolExecutor() as executor:
    filenames = [f for f in os.listdir(directory) if f.endswith('.csv')]
    list(tqdm(executor.map(process_file, filenames), total=len(filenames), desc="Processing"))
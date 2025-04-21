import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'results/reconstruction_loss/csv/test_1.csv'
df = pd.read_csv(file_path)

df.hist(bins=10000)
plt.tight_layout()
plt.title('1/17')
plt.show()

df.plot()
plt.show()

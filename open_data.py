import numpy as np
import pandas as pd
file_path='embeddings.npy'

a=np.load(file_path)
for i in range(0,4):
    print(a[i])
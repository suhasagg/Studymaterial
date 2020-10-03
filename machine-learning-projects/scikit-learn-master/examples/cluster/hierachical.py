# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
df = pd.read_csv("Prestige.csv")
labels = np.array(df['type'])
df = df.iloc[:, 0:4]
df.head()

Z = linkage(df, 'ward')
plt.figure(figsize=(50, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Prestige Dataset')
plt.ylabel('distance')
dendrogram(
    Z,
    labels=labels,
    leaf_rotation=90.,
    leaf_font_size=8.,
)
plt.show()
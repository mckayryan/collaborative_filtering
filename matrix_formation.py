
# coding: utf-8

# In[6]:

import warnings
warnings.filterwarnings('ignore')

import numpy as npy
import pandas as pd
from numpy import linalg as lla


# In[8]:
# must change this path 
# movies_data = pd.read_csv('/home/ryan/uni/machine_learning/project/data/ml-latest-small/%s.csv' % 'ratings')
movies_df = pd.DataFrame(movies_data)


# In[ ]:




# In[9]:

movies_df.set_index(['userId'])
movies_df.drop(['timestamp'], axis=1)


# In[13]:

movies_df.pivot(index='userId', columns='movieId', values='rating')
movies_df.as_matrix()


# In[ ]:

U,s,V = lla.svd(movies_df, full_matricies=True)


# In[ ]:




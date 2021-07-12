#!/usr/bin/env python
# coding: utf-8

# In[2]:

# install CUDA

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "C:/Users/LENOVO/Downloads/kvasir-dataset/"
CATEGORIES = ["dyed-lifted-polyps","esophagitis","dyed-resection-margins","normal-cecum","normal-pylorus","normal-z-line", "polyps","ulcerative-colitis"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break

# In[4]:


print(img_array.shape)

# In[11]:


IMG_SIZE = 200

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#plt.imshow(new_array, cmap="gray")
#plt.show()

# In[13]:


training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()

print(len(training_data))


random.shuffle(training_data)

# In[17]:


#for sample in training_data[:10]:
#    print(sample[1])

# In[19]:


X = []
y = []

# In[21]:


for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE,1) #,1

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# In[ ]:
#!/usr/bin/env python
# coding: utf-8

# In[1]:






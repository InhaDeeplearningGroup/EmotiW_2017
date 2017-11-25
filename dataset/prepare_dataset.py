__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-
import os
from skimage.measure import block_reduce
import cv2
import numpy as np
from six.moves import xrange
from skimage.io import imread

def crop_and_resize(img, target_size=32, zoom=1):
    small_side = int(np.min(img.shape) * zoom)
    reduce_factor = small_side / target_size
    crop_size = target_size * reduce_factor
    mid = np.array(img.shape) / 2
    half_crop = crop_size / 2
    center = img[mid[0]-half_crop:mid[0]+half_crop,
    	mid[1]-half_crop:mid[1]+half_crop]
    return block_reduce(center, (reduce_factor, reduce_factor), np.mean)

def list_all_files(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            joined = os.path.join(root, filename)
            if extensions is None or ext.lower() in extensions:
                yield joined
                
def list_box(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        boxes = []
        boxes.append(dirnames)
        for box in boxes:
            return box
        
def box_paths(directory, box):
    joined = []
    for i in xrange(len(box)):
        joined.append(os.path.join(directory, str(box[i])))
    return joined

def to_dataset(examples):
    X = []
    y = []
    X_zero = np.zeros(shape=(224, 224))
    for i in xrange(len(examples)):
        j = 0
        for path, label in examples[i]:
            img = imread(path, as_grey=True)
            img = cv2.resize(img, (224, 224)) # ImageNet Size
            img_h = cv2.equalizeHist(img) # Histogram Equalization
            X.append(img_h)
            y.append(label)
            j = j + 1
            if j >= 40:
                break
        if j < 40:
            for w in xrange(40-j):
                X.append(X_zero)
    return np.asarray(X), np.asarray(y)

# %% load path of facial picture

angry_box = list(list_box('/home/kdh/Desktop/AFEW_MY/Train/Angry/', ['.jpg', '.png']))
angry_paths = box_paths('/home/kdh/Desktop/AFEW_MY/Train/Angry/', angry_box)
a_angry = []
for i in xrange(len(angry_paths)):
    angry_path = list(list_all_files(angry_paths[i], ['.jpg', '.png']))
    a_angry.append(angry_path)
    angry_path = []
print('loaded', len(a_angry), 'final angry file lists')

disgust_box = list(list_box('/home/kdh/Desktop/AFEW_MY/Train/Disgust/', ['.jpg', '.png']))
disgust_paths = box_paths('/home/kdh/Desktop/AFEW_MY/Train/Disgust/', disgust_box)
a_disgust = []
for i in xrange(len(disgust_paths)):
    disgust_path = list(list_all_files(disgust_paths[i], ['.jpg', '.png']))
    a_disgust.append(disgust_path)
    disgust_path = []
print('loaded', len(a_disgust), 'final angry file lists')

fear_box = list(list_box('/home/kdh/Desktop/AFEW_MY/Train/Fear/', ['.jpg', '.png']))
fear_paths = box_paths('/home/kdh/Desktop/AFEW_MY/Train/Fear/', fear_box)
a_fear = []
for i in xrange(len(fear_paths)):
    fear_path = list(list_all_files(fear_paths[i], ['.jpg', '.png']))
    a_fear.append(fear_path)
    fear_path = []
print('loaded', len(a_fear), 'final angry file lists')

happy_box = list(list_box('/home/kdh/Desktop/AFEW_MY/Train/Happy/', ['.jpg', '.png']))
happy_paths = box_paths('/home/kdh/Desktop/AFEW_MY/Train/Happy/', happy_box)
a_happy = []
for i in xrange(len(happy_paths)):
    happy_path = list(list_all_files(happy_paths[i], ['.jpg', '.png']))
    a_happy.append(happy_path)
    happy_path = []
print('loaded', len(a_happy), 'final angry file lists')

neutral_box = list(list_box('/home/kdh/Desktop/AFEW_MY/Train/Neutral/', ['.jpg', '.png']))
neutral_paths = box_paths('/home/kdh/Desktop/AFEW_MY/Train/Neutral/', neutral_box)
a_neutral = []
for i in xrange(len(neutral_paths)):
    neutral_path = list(list_all_files(neutral_paths[i], ['.jpg', '.png']))
    a_neutral.append(neutral_path)
    neutral_path = []
print('loaded', len(a_neutral), 'final angry file lists')

sad_box = list(list_box('/home/kdh/Desktop/AFEW_MY/Train/Sad/', ['.jpg', '.png']))
sad_paths = box_paths('/home/kdh/Desktop/AFEW_MY/Train/Sad/', sad_box)
a_sad = []
for i in xrange(len(sad_paths)):
    sad_path = list(list_all_files(sad_paths[i], ['.jpg', '.png']))
    a_sad.append(sad_path)
    sad_path = []
print('loaded', len(a_sad), 'final angry file lists')

surprise_box = list(list_box('/home/kdh/Desktop/AFEW_MY/Train/Surprise/', ['.jpg', '.png']))
surprise_paths = box_paths('/home/kdh/Desktop/AFEW_MY/Train/Surprise/', surprise_box)
a_surprise = []
for i in xrange(len(surprise_paths)):
    surprise_path = list(list_all_files(surprise_paths[i], ['.jpg', '.png']))
    a_surprise.append(surprise_path)
    surprise_path = []
print('loaded', len(a_surprise), 'final angry file lists')

# %% labeling process

angry_examples = []
for i in xrange(len(a_angry)):
    angry_examples.append([(path, 0) for path in a_angry[i]])
    
disgust_examples = []
for i in xrange(len(a_disgust)):
    disgust_examples.append([(path, 1) for path in a_disgust[i]])

fear_examples = []
for i in xrange(len(a_fear)):
    fear_examples.append([(path, 2) for path in a_fear[i]])
    
happy_examples = []
for i in xrange(len(a_happy)):
    happy_examples.append([(path, 3) for path in a_happy[i]])
    
neutral_examples = []
for i in xrange(len(a_neutral)):
    neutral_examples.append([(path, 4) for path in a_neutral[i]])
    
sad_examples = []
for i in xrange(len(a_sad)):
    sad_examples.append([(path, 5) for path in a_sad[i]])
    
surprise_examples = []
for i in xrange(len(a_surprise)):
    surprise_examples.append([(path, 6) for path in a_surprise[i]])



#%% load dataset from path

X_angry, _ = to_dataset(angry_examples)
X_disgust, _ = to_dataset(disgust_examples)
X_fear, _ = to_dataset(fear_examples)
X_happy, _ = to_dataset(happy_examples)
X_neutral, _ = to_dataset(neutral_examples)
X_sad, _ = to_dataset(sad_examples)
X_surprise, _ = to_dataset(surprise_examples)

X_train_list = []
X_train_list.append(X_angry)
X_train_list.append(X_disgust)
X_train_list.append(X_fear)
X_train_list.append(X_happy)
X_train_list.append(X_neutral)
X_train_list.append(X_sad)
X_train_list.append(X_surprise)
X_train = np.vstack(X_train_list) ## finish

y_angry = np.zeros(shape=(len(X_angry), 1))
y_disgust = np.zeros(shape=(len(X_disgust), 1))
for i in xrange(len(y_disgust)):
    y_disgust[i,:] = 1
y_fear = np.zeros(shape=(len(X_fear), 1))
for i in xrange(len(y_fear)):
    y_fear[i,:] = 2
y_happy = np.zeros(shape=(len(X_happy), 1))
for i in xrange(len(y_happy)):
    y_happy[i,:] = 3
y_neutral = np.zeros(shape=(len(X_neutral), 1))
for i in xrange(len(y_neutral)):
    y_neutral[i,:] = 4
y_sad = np.zeros(shape=(len(X_sad), 1))
for i in xrange(len(y_sad)):
    y_sad[i,:] = 5
y_surprise = np.zeros(shape=(len(X_surprise), 1))
for i in xrange(len(y_surprise)):
    y_surprise[i,:] = 6

y_train_list = []
y_train_list.append(y_angry)
y_train_list.append(y_disgust)
y_train_list.append(y_fear)
y_train_list.append(y_happy)
y_train_list.append(y_neutral)
y_train_list.append(y_sad)
y_train_list.append(y_surprise)
y_train = np.vstack(y_train_list) ## finish

                   
#%% In case of S3DAE and C3DA

X_angry_frame = len(X_angry) // 40
X_disgust_frame = len(X_disgust) // 40
X_fear_frame = len(X_fear) // 40
X_happy_frame = len(X_happy) // 40
X_neutral_frame = len(X_neutral) // 40
X_sad_frame = len(X_sad) // 40
X_surprise_frame = len(X_surprise) // 40    
                      
y_angry_frame = np.zeros(shape=(X_angry_frame, 1))
y_disgust_frame = np.zeros(shape=(len(X_disgust_frame), 1))
for i in xrange(len(y_disgust_frame)):
    y_disgust_frame[i,:] = 1
y_fear_frame = np.zeros(shape=(len(X_fear_frame), 1))
for i in xrange(len(y_fear_frame)):
    y_fear_frame[i,:] = 2
y_happy_frame = np.zeros(shape=(len(X_happy_frame), 1))
for i in xrange(len(y_happy_frame)):
    y_happy_frame[i,:] = 3
y_neutral_frame = np.zeros(shape=(len(X_neutral_frame), 1))
for i in xrange(len(y_neutral_frame)):
    y_neutral_frame[i,:] = 4
y_sad_frame = np.zeros(shape=(len(X_sad_frame), 1))
for i in xrange(len(y_sad_frame)):
    y_sad_frame[i,:] = 5
y_surprise_frame = np.zeros(shape=(len(X_surprise_frame), 1))
for i in xrange(len(y_surprise_frame)):
    y_surprise_frame[i,:] = 6

y_train_frame_list = []
y_train_list.append(y_angry_frame)
y_train_list.append(y_disgust_frame)
y_train_list.append(y_fear_frame)
y_train_list.append(y_happy_frame)
y_train_list.append(y_neutral_frame)
y_train_list.append(y_sad_frame)
y_train_list.append(y_surprise_frame)
y_train_frame = np.vstack(y_train_frame_list) ## finish


#%% save as npz format

np.savez('/home/kdh/AFEW/X_train.npz', X_train)
np.savez('/home/kdh/AFEW/y_train.npz', y_train)
np.savez('/home/kdh/AFEW/y_frmae.npz', y_train_frame)




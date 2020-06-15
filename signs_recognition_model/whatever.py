#!/usr/bin/env python3

import os
import sys
from fastai.vision import *

image_path = sys.argv[1]
path = Path('/Users/buena/Desktop/signs')

defaults.device = torch.device('cpu')
img = open_image(image_path)

learn = load_learner(path)
pred_class,pred_idx,outputs = learn.predict(img)
print(pred_class)

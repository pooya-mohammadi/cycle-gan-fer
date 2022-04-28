# cycle-gan-fer
Cycle GAN is one of the renowned models for generating solving image-to-image translation for an un-paired dataset. In this repository we aim to provide
a cycle gan model for generating new samples for FER-2013(facial expression recognition) dataset. The aforementioned dataset is biased which means that the number of samples in classes are the same or even close together. 
To tackle this issue we utilize cycle-gan model to generate new samples for `disgust` class which has the lowest number of samples.  

## Train 
To train the model run the following command:
```commandline
python train.py
```
The parameters are listed in `config.py` model:
``` python
IMG_SAVE_INTERVAL = 5
BATCH_SIZE = 1
GEN_LEARNING_RATE = 1e-5
DIS_LEARNING_RATE = 5e-6
LAMBDA_IDENTITY = 0.0
CYCLE_LOSS_COEFFICIENT = 5
LAMBDA_GEN_IDENTITY = 1.0
...
```

## References
Credit goes to Aladdin Persson [Youtube](https://www.youtube.com/watch?v=4LktBHGCNfw) [Github](https://github.com/aladdinpersson/Machine-Learning-Collection)


import numpy as np
import pandas as pd


def write_values(values,path):
    with open(path,'w') as f:
        for ar in values:
            f.write(ar+'\n')


data_path='data/'
train_path=data_path+'AutoMaster_TrainSet.csv'


ARTICLE_FILE = data_path+"train_text.txt"
SUMMARRY_FILE = data_path+"train_label.txt"


train=pd.read_csv(train_path)

train=train.dropna()

train['merged'] = train[['Question', 'Dialogue']].apply(lambda x: ' '.join(x),axis=1)

write_values(train['merged'].values,ARTICLE_FILE)
write_values(train['Report'].values,SUMMARRY_FILE)


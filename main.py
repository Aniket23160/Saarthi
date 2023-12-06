
import spacy
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
from scipy.spatial import distance
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from simpletransformers.ner import NERModel,NERArgs
import os
import random 
import yaml
import argparse
import pickle
import tensorflow as tf
import datetime


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    default="/home/aniket/Saarthi/config.yaml",
    type=str,
    help="path to config file.",
)
args = parser.parse_args()

# Load config yaml file 
cfg = yaml.safe_load(open(args.config_file, "r"))


os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = SentenceTransformer('distilbert-base-nli-mean-tokens')


def similar(word1,word2):
    words=[word1,word2]
    word_embeddings=model.encode(words)
    return (1 - distance.cosine(word_embeddings[0], word_embeddings[1]))

with open(cfg['VALIDATION_DATA_PATH']) as file_obj:

    reader_obj = csv.DictReader(file_obj)
    action,object,location='','',''
    action_val,object_val,location_val=0,0,0
    count=1
    temp=[]
    for row in reader_obj:
        print(count)
        d=row['transcription'].split()
        for word in d:
            sim=similar(word,row['action'])
            if sim>action_val:
                action_val=sim
                action=word
            sim=similar(word,row['object'])
            if sim>action_val:
                object_val=sim
                object=word
            sim=similar(word,row['location'])
            if sim>action_val:
                location_val=sim
                location=word
        for i in range(len(d)-1):
            word=d[i]+'_'+d[i+1]
            d.append(d[i]+'_'+d[i+1])
            sim=similar(word,row['action'])
            if sim>action_val:
                action_val=sim
                action=word
            sim=similar(word,row['object'])
            if sim>action_val:
                object_val=sim
                object=word
            sim=similar(word,row['location'])
            if sim>action_val:
                location_val=sim
                location=word

        for word in d:
            if word==action:
                temp.append([count,word,'action'])
            elif word==object:
                temp.append([count,word,'object'])
            elif word==location:
                temp.append([count,word,'location'])
            else:
                temp.append([count,word,'O'])
        count+=1

with open(cfg['MODIFIED_SAVE_AT']['TRAIN'],"w+") as my_csv:
    newarray = csv.writer(my_csv,delimiter=',')
    newarray.writerows(temp)

data = pd.read_csv(cfg['MODIFIED_SAVE_AT']['TRAIN'],encoding="latin1" )

data.rename(columns={"1":"sentence_id","Turn":"words","O":"labels"}, inplace =True)
data["labels"] = data["labels"].str.upper()
X= data[["sentence_id","words"]]
Y =data["labels"]
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size =0.2)

train_data = pd.DataFrame({"sentence_id":x_train["sentence_id"],"words":x_train["words"],"labels":y_train})
test_data = pd.DataFrame({"sentence_id":x_test["sentence_id"],"words":x_test["words"],"labels":y_test})


log_dir = cfg['OUTPUT_DIR'] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
label = data["labels"].unique().tolist()

# Model Parameters
args = NERArgs()
args.num_train_epochs = cfg['NUM_TRAIN_EPOCHS']
args.learning_rate = 1e-4
args.overwrite_output_dir =cfg['OVERWRITE_OUTPUT_DIR']
args.train_batch_size = cfg['TRAIN_BATCH_SIZE']
args.eval_batch_size = cfg['EVAL_BATCH_SIZE']
model = NERModel('bert', 'bert-base-cased',labels=label,args =args)

# Train the model
model.train_model(train_data,eval_data = test_data,acc=accuracy_score,callbacks=[tensorboard])
result, model_outputs, preds_list = model.eval_model(test_data)

# save the model as a pickle file
model_pkl_file = cfg['MODEL_SAVE_AT']

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)

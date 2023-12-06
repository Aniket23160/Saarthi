
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="/home/aniket/Saarthi/config.yaml",
        type=str,
        help="path to config file.",
    )
    args = parser.parse_args()

    # Load config yaml file as nested object
    cfg = yaml.safe_load(open(args.config_file, "r"))
    print(cfg)

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

    with open(cfg['MODIFIED_SAVE_AT']['TEST'],"w+") as my_csv:
        newarray = csv.writer(my_csv,delimiter=',')
        newarray.writerows(temp)

    data = pd.read_csv(cfg['MODIFIED_SAVE_AT']['TEST'],encoding="latin1" )

    data.rename(columns={"1":"sentence_id","Turn":"words","O":"labels"}, inplace =True)
    data["labels"] = data["labels"].str.upper()
    X= data[["sentence_id","words"]]
    Y =data["labels"]

    test_data = pd.DataFrame({"sentence_id":X["sentence_id"],"words":X["words"],"labels":Y})

    #Calling the saved model
    model_pkl_file = cfg['MODEL_SAVE_AT']  

    #Running Evaluation Task
    with open(model_pkl_file, 'rb') as file:  
        model = pickle.load(file)
        result, model_outputs, preds_list = model.eval_model(test_data)
        print(result)
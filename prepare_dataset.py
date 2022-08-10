from fileinput import filename
from unicodedata import category
import librosa
import os
import json

dataset_path="data"
json_path="data.json"
samples=22050

def prepare_dataset(path,json_path,n_mfcc=13, hop_length=512,n_fft=2048):
    #data dictionary
    data={
        "mappings":[],
        "labels":[],
        "MFCCs":[],
        "files":[]
    }
    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(path)):
        
        if dirpath is not path:
            #update mappings
            category=dirpath.split("/")[-1]  #dataset/down -> [dataset,down]
            data["mappings"].append(category)
            print(f"processing :{category}")

            #loop through all filenames and extract its mfcc
            for f in filenames:

                # get file path
                file_path=os.path.join(dirpath,f)

                # load audio file
                signal,sr=librosa.load(file_path)

                # ensure audio file is atleast 1 sec
                if len(signal)>=samples:
                    #enforce 1 sec
                    signal=signal[:samples]

                    # extract mfcc
                    mfcc=librosa.feature.mfcc(signal,sr=sr,n_mfcc=n_mfcc,n_fft=n_fft)

                    # store data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(mfcc.T.tolist())
                    data["files"].append(file_path)
                    print(f'{file_path}:{i-1}')

    #store in json format
    with open(json_path,"w") as fp:
        json.dump(data,fp,indent=4)

# if __name__=="main":
prepare_dataset(dataset_path,json_path)

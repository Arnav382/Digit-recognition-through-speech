"""
client->POST request->server->prediction->back to client
"""

import os
from keyboard_spotting_service import keyword_spotting_service
import random

from flask import Flask,request,jsonify

app=Flask(__name__)

"""
ks.com/predict
"""
@app.route("/predict",methods=["POST"])
def predict():
    # get audio file and save it
    audio_file=request.files["file"]
    file_name=str(random.randint(0,1000))
    audio_file.save(file_name)

    # invoke keyword spotting service
    kss=keyword_spotting_service()
    
    # make prediction
    predicted_keyword=kss.predict(file_name)

    # remove the audio file
    os.remove(file_name)

    # send back the predicted word to client in json format
    data={"keyword ":predicted_keyword}
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=False)
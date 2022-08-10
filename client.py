from urllib import response
import requests

# run this file from the terminal, hence full path is given for the audio files
URL="http://127.0.0.1:5000/predict"
TEST_AUDIO_FILE_PATH="/Users/arnavbhattacharya/Desktop/Projects/NLP/Speech to text/test/5.wav"

if __name__=="__main__":
    audio_file=open(TEST_AUDIO_FILE_PATH,"rb")
    values={"file": (TEST_AUDIO_FILE_PATH,audio_file, "audio/wav")}
    response=requests.post(URL,files=values)
    data=response.json()
    print(f"predicted keyword is : {data['keyword']}")
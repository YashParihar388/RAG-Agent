import whisper
import os

whisper_model = os.getenv("WHISPER_API_KEY","small")

_model = None

def load_model():
    if _model is None:
        print("loading model ...")
        _model = whisper.load_model(whisper_model)
        print("model loaded successfully")
        
    return _model

def process_input(chunk_path: str,translate: bool = False):
    model = load_model()
    task = "translate" if translate else "transcribe"
    
    result = model.transcribe(chunk_path,task)
    
    return result['text']

def transcribe_all(chunks:list,translate: bool = False):
    full_transcription = ""
    
    for i,chunk in enumerate(chunks):
        print(f"transcribing chunk {i+1}")
        text = process_input(chunk,translate = translate)
        
        full_transcription+=text + " "
    
    print("transcription completed")
    return full_transcription
        


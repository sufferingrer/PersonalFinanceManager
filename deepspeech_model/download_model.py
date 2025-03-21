import os
import tarfile
import urllib.request

def download_deepspeech_model():
    """
    Downloads and extracts the DeepSpeech pre-trained model.
    The model files will be downloaded to a 'deepspeech_model' directory.
    """
    model_url = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.tflite"
    scorer_url = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"
    
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "model.tflite")
    scorer_path = os.path.join(model_dir, "model.scorer")
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Download model and scorer if they don't exist
    if not os.path.exists(model_path):
        print("Downloading DeepSpeech model...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Model downloaded to {model_path}")
    
    if not os.path.exists(scorer_path):
        print("Downloading DeepSpeech scorer...")
        urllib.request.urlretrieve(scorer_url, scorer_path)
        print(f"Scorer downloaded to {scorer_path}")
    
    return model_path, scorer_path

if __name__ == "__main__":
    download_deepspeech_model()

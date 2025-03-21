import io
import numpy as np
import wave
import os
from gtts import gTTS
import sounddevice as sd
import soundfile as sf
import tempfile
import speech_recognition as sr
from typing import List, Dict, Any, Optional, Tuple, Union

class AudioProcessor:
    """
    Utility class for audio processing, including:
    - Text-to-Speech using gTTS
    - Speech-to-Text using SpeechRecognition
    - Audio playback
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust recognition parameters
        self.recognizer.energy_threshold = 300  # Minimum audio energy to consider for recording
        self.recognizer.dynamic_energy_threshold = True  # Automatically adjust for ambient noise
        self.recognizer.pause_threshold = 0.8  # Seconds of non-speaking audio before a phrase is considered complete
    
    def text_to_speech(self, text: str, lang: str = 'ru') -> Tuple[bytes, int]:
        """
        Convert text to speech using gTTS
        
        Args:
            text: Text to convert to speech
            lang: Language code (default: 'ru' for Russian)
            
        Returns:
            Tuple of (audio_bytes, sample_rate)
        """
        # Use gTTS to generate audio
        tts = gTTS(text=text, lang=lang, slow=False)
        
        # Save to a temporary BytesIO object
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        # Convert to format compatible with sounddevice
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_file.write(audio_bytes.read())
            temp_path = temp_file.name
            
        try:
            # Read the audio file using soundfile
            data, sample_rate = sf.read(temp_path)
            
            # Convert to WAV bytes
            wav_bytes = io.BytesIO()
            with wave.open(wav_bytes, 'wb') as wf:
                wf.setnchannels(1 if len(data.shape) == 1 else data.shape[1])
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes((data * 32767).astype(np.int16).tobytes())
            
            return wav_bytes.getvalue(), sample_rate
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def play_audio(self, audio_data: bytes, sample_rate: int = 16000):
        """
        Play audio from bytes
        
        Args:
            audio_data: Audio data as bytes
            sample_rate: Sample rate of the audio
        """
        with io.BytesIO(audio_data) as wav_file:
            with wave.open(wav_file, 'rb') as wf:
                # Get audio parameters
                channels = wf.getnchannels()
                width = wf.getsampwidth()
                rate = wf.getframerate() if sample_rate is None else sample_rate
                
                # Read audio data
                frames = wf.readframes(wf.getnframes())
                
                # Convert to numpy array for sounddevice
                dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
                audio_array = np.frombuffer(frames, dtype=dtype_map.get(width, np.int16))
                
                # Reshape for stereo
                if channels == 2:
                    audio_array = audio_array.reshape(-1, 2)
                
                # Play audio
                sd.play(audio_array, rate)
                sd.wait()
    
    def speech_to_text(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Convert speech to text using SpeechRecognition
        
        Args:
            audio_bytes: Audio data as bytes
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        try:
            # Create AudioData from WAV bytes
            with io.BytesIO(audio_bytes) as audio_file:
                audio_data = sr.AudioData(audio_file.read(), sample_rate=sample_rate, sample_width=2)
                
                # Use Google's API for speech recognition (requires internet)
                text = self.recognizer.recognize_google(audio_data, language="ru-RU")
                return text
        except sr.UnknownValueError:
            return "Речь не распознана. Пожалуйста, говорите чётче."
        except sr.RequestError as e:
            return f"Ошибка сервиса распознавания речи: {e}"
        except Exception as e:
            print(f"Error in speech to text conversion: {str(e)}")
            return "Ошибка распознавания речи"

    def frames_to_wav_bytes(self, frames: List[np.ndarray], sample_rate: int = 16000) -> bytes:
        """
        Convert a list of audio frames to WAV bytes
        
        Args:
            frames: List of numpy arrays containing audio data
            sample_rate: Sample rate of the audio
            
        Returns:
            WAV audio as bytes
        """
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                
                # Concatenate and write frames
                if frames:
                    all_frames = np.concatenate(frames, axis=0)
                    wf.writeframes(all_frames.tobytes())
                
            return wav_buffer.getvalue()

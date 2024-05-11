# AUTOMATIC SPEECH RECOGNITION

## OVERVIEW
Speech recognition, also known as automatic speech recognition (ASR), computer speech recognition or speech-to-text, is a capability that enables a program to process human speech into a written format.
While speech recognition is commonly confused with voice recognition, speech recognition focuses on the translation of speech from a verbal format to a text one whereas voice recognition just seeks to identify an individual user’s voice. (https://www.ibm.com/topics/speech-recognition#:~:text=Speech%20recognition%2C%20also%20known%20as,speech%20into%20a%20written%20format.)


## DATASET
The dataset used is minds14 (https://huggingface.co/datasets/PolyAI/minds14).  
This dataset has 8168 audio and true transcrptions pair.

| ![Minds-14 Preview](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/dataset_preview.PNG) |
|:--:| 
| The preview of the Minds-14 dataset |

### Audio Length
In general, the audio array in this dataset has 
- minimum length of 10K 
- average length of 80K
- maximum length of 500K


| ![The length of the audio](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/length_audio_array.png) |
|:--:| 
| The length of the audio |

### Sampling Rate
The sampling rate of the audio data is 8000 Hz


### Language and Intent Class
The audio data in the Minds-14 dataset consists of various language and intent.

| ![The language composition](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/language_count.png) |
|:--:| 
| The language composition |

| ![The intent class](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/intent_count.png) |
|:--:| 
| The intent class |

| ![The intent class with respect to the language of the audio](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/language_intent_count.png) |
|:--:| 
| The intent class with respect to the language of the audio |


## AUDIO ANALYSIS
When dealing with audio data, several audio processing techniques can be employed, e.g. spectrogram, etc, which is used as the input feature for the model later.  
Some of these convert the audio data to the frequency domain.


| ![Spectrogram of the audio represents the signal strength (loudness) of a signal over time at various frequencies present in a particular waveform](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/spectrogram.png) |
|:--:| 
| Spectrogram of the audio represents the signal strength (loudness) of a signal over time at various frequencies present in a particular waveform |


| ![Spectrogram in logarithmic scale shows how the loudness of the audio is at the lower frequency range](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/log_spectrogram.png) |
|:--:| 
| Spectrogram in logarithmic scale shows how the loudness of the audio is at the lower frequency range |



| ![MFCC of the audio represents the short-term power spectrum of sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/mfcc.png) |
|:--:| 
| MFCC of the audio represents the short-term power spectrum of sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency |



| ![Chroma, also known as chroma vectors or chromagrams, are powerful tools in the field of music information retrieval, used to represent the intensity of the twelve different pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B) regardless of the octave in which they appear](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/chroma.png) |
|:--:| 
| Chroma, also known as chroma vectors or chromagrams, are powerful tools in the field of music information retrieval, used to represent the intensity of the twelve different pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B) regardless of the octave in which they appear |


## PREPROCESSING
For simplicity, only the English audio is used for this task.

The preprocessing is as following
1. Train - test split: 90-10 ratio
2. The audio data is preprocessed following the pretrained model's configuration


The evaluation uses Word Error Rate (WER) and accuracy.
The calculation for the accuracy is: 


## TRANSCRIPTION

### Whisper V3 Large
https://huggingface.co/openai/whisper-large-v3

Number of parameters: 1.54B parameters

Whisper V3 is trained on dataset with sampling rate of 16K Hz

Result:
| WER     | Accuracy |
| -------- | ----- |
| 0.27  | 72.95%    |


| ![Transcription result using Whisper V3 Large](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/whisper_v3_result.PNG) |
|:--:| 
| Transcription result using Whisper V3 Large |



### Wav2vec2 Large
https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english

Number of parameters: 315M parameters

Wav2vec2 is trained on dataset with sampling rate of 16K Hz

Result:
| WER     | Accuracy |
| -------- | ----- |
| 0.349  | 65.09%    |


| ![Transcription result using Wav2vec2 Large](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/wav2vev2_large_result.PNG) |
|:--:| 
| Transcription result using Wav2vec2 Large |



### Wav2vec2 Base
https://huggingface.co/facebook/wav2vec2-base-960h

Number of parameters: 94.4M parameters

Wav2vec2 is trained on dataset with sampling rate of 16K Hz


Result:
| WER     | Accuracy |
| -------- | ----- |
| 0.57  | 42.19%    |


| ![Transcription result using Wav2vec2 Base](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Automatic-Speech-Recognition/images/wav2vev2_base_result.PNG) |
|:--:| 
| Transcription result using Wav2vec2 Base |


## CONCLUSION
1. Among three pretrained models explored, the best result is using Whisper V3 Large, which achieves accuracy score of 72.95% and WER of 0.27
2. The transcription generated by each model is able to capture the key words in the audio, although not perfectly fits to the true transcription.

Improvement idea:
1. The sampling rate of the pretrained models is 16k Hz, whereas the sampling rate of the audio in the Minds-14 dataset is 8k Hz. Upsampling the audio may improve the result
2. Fine tune the pretrained model on the Minds-14 dataset


This automatic speech recognition task is part of the projects done during NLP bootcamp at Indonesia AI.
Other projects during the NLP bootcamp can be found at https://github.com/RobyKoeswojo/Indonesia-AI

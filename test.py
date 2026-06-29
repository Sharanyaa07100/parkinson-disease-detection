import numpy as np
import os
import pandas as pd
import parselmouth
from parselmouth.praat import call

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)#create a praat pitch object
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity05 = call(sound, "To Harmonicity (cc)", 0.01, 500, 0.1, 1.0)
    hnr05 = call(harmonicity05, "Get mean", 0, 0)
    harmonicity15 = call(sound, "To Harmonicity (cc)", 0.01, 1500, 0.1, 1.0)
    hnr15 = call(harmonicity15, "Get mean", 0, 0)
    harmonicity25 = call(sound, "To Harmonicity (cc)", 0.01, 2500, 0.1, 1.0)
    hnr25 = call(harmonicity25, "Get mean", 0, 0)
    harmonicity35 = call(sound, "To Harmonicity (cc)", 0.01, 3500, 0.1, 1.0)
    hnr35 = call(harmonicity35, "Get mean", 0, 0)
    harmonicity38 = call(sound, "To Harmonicity (cc)", 0.01, 3800, 0.1, 1.0)
    hnr38 = call(harmonicity38, "Get mean", 0, 0)
    return [localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38]

def getLabel(name):
    label = 0
    if name == "PD":
        label = 1
    return label    

path = "ParkinsonDataset/ReadText"
X = []
Y = []
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        label = getLabel(name)
        sound = parselmouth.Sound(root+"/"+directory[j])
        features = measurePitch(sound, 75, 1000, "Hertz")
        X.append(features)
        Y.append(label)
        print(name+" "+str(label)+" "+root+"/"+directory[j]+" "+str(features))

path = "ParkinsonDataset/SpontaneousDialogue"
for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        label = getLabel(name)
        sound = parselmouth.Sound(root+"/"+directory[j])
        features = measurePitch(sound, 75, 1000, "Hertz")
        X.append(features)
        Y.append(label)
        print(name+" "+str(label)+" "+root+"/"+directory[j]+" "+str(features))        
        


dataset = pd.DataFrame(X, columns=["Jitter_rel","Jitter_abs","Jitter_RAP","Jitter_PPQ","Shim_loc","Shim_dB","Shim_APQ3","Shim_APQ5","Shi_APQ11",
                                "hnr05", "hnr15", "hnr25", "hnr35", "hnr38"])

dataset['hnr25'].fillna((dataset['hnr25'].mean()), inplace=True) #Data cleaning because they may be NaN values
dataset['hnr15'].fillna((dataset['hnr15'].mean()), inplace=True) #Data cleaning because they may be NaN values
dataset['hnr35'].fillna((dataset['hnr35'].mean()), inplace=True) #Data cleaning because they may be NaN values
dataset['hnr38'].fillna((dataset['hnr38'].mean()), inplace=True) #Data cleaning because they may be NaN values
dataset['Label'] = Y
dataset.to_csv("ProcessedData/processed_results.csv", index=False)



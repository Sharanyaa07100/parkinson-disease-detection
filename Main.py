from tkinter import *
from tkinter.filedialog import askopenfilename
import tkinter
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import model_from_json
import pickle
import os
import parselmouth
from parselmouth.praat import call
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


gui = tkinter.Tk()
gui.title("Parkinson Disease Detection Using Deep Neural Networks")
gui.geometry("1300x1200")

global filename
global X_train, X_test, y_train, y_test, cnn
global dataset, X, Y
sc = MinMaxScaler(feature_range = (0, 1))

def getLabel(name):
    label = 0
    if name == "PD":
        label = 1
    return label   

def measurePitch(voice_data, f0min, f0max, unit):
    sound = parselmouth.Sound(voice_data) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)#create a praat pitch object
    #extracting the features from the point process sound
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
    #exracted features from sound
    return [localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, hnr05, hnr15 ,hnr25 ,hnr35 ,hnr38]


def uploadDataset():
    global filename, dataset, X, Y
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    output.delete('1.0', END)
    output.insert(END,filename+" loaded\n\n")
    if os.path.exists("ProcessedData/processed_results.csv"):
        dataset = pd.read_csv("ProcessedData/processed_results.csv")
        dataset.fillna(0, inplace = True)
    else:
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
        dataset = pd.read_csv("ProcessedData/processed_results.csv")
        dataset.fillna(0, inplace = True)
    output.insert(END,"Extracted features from speech files\n\n")    
    output.insert(END,str(dataset.head())+"\n\n")
    output.update_idletasks()
    diseaseGraph = dataset.groupby('Label').size()
    diseaseGraph.plot(kind="bar")
    plt.title("Health & Parkinson Disease Graph 0 (Healthy) & 1 (Parkinson Disease)")
    plt.show()       
    

def preprocessDataset():
    output.delete('1.0', END)
    global X_train, X_test, y_train, y_test, dataset
    global X, Y, sc
    dataset = dataset.values 
    X = dataset[:,0:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    sc = MinMaxScaler() 
    sc.fit(X)
    X = sc.transform(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    output.insert(END,"Total Speech Files found in dataset : "+str(X.shape[0])+"\n\n")
    output.insert(END,"Dataset train & test split details. 80% dataset records used for training & 20% used for testing\n\n")
    output.insert(END,"Training Size : "+str(X_train.shape[0])+"\n")
    output.insert(END,"Training Size : "+str(X_test.shape[0])+"\n\n")
    

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    output.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    output.insert(END,algorithm+" Precision : "+str(p)+"\n")
    output.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    output.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(y_test, predict)
    labels = ['Healthy', 'Parkinson Disease']
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def trainSVM():
    global accuracy, precision, recall, fscore
    global X_train, X_test, y_train, y_test, X, Y
    output.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    svm_cls = svm.SVC() 
    svm_cls.fit(X, Y) 
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)
    
def trainXGBoost():
    xgb_cls = xgb.XGBClassifier() 
    xgb_cls.fit(X_train, y_train) 
    predict = xgb_cls.predict(X_test)
    calculateMetrics("XGBoost", predict, y_test)
    
def trainMLP():
    mlp_cls = MLPClassifier(max_iter=5000) 
    mlp_cls.fit(X_train, y_train) 
    predict = mlp_cls.predict(X_test)
    calculateMetrics("MLP", predict, y_test)


def trainCNN():
    global X, Y, cnn
    X1 = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    Y1 = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_cnn_json = json_file.read()
            cnn = model_from_json(loaded_cnn_json)
        json_file.close()
        cnn.load_weights("model/model_weights.h5")
        cnn._make_predict_function()
    else:
        cnn = Sequential()
        cnn.add(Convolution2D(128, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (1, 1)))
        cnn.add(Convolution2D(256, 1, 1, activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (1, 1)))
        cnn.add(Flatten())
        cnn.add(Dense(output_dim = 256, activation = 'relu'))
        cnn.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
        cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = cnn.fit(X_train, y_train, batch_size=4, epochs=30, shuffle=True, verbose=2, validation_data=(X_test, y_test))
        cnn.save_weights('model/model_weights.h5')            
        model_json = cnn.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
    predict = cnn.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    predict = np.argmax(predict, axis=1)
    calculateMetrics("CNN", predict, y_test)


def predictDisease():
    global cnn, sc
    output.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testSpeechFiles")
    sound = parselmouth.Sound(filename)
    features = measurePitch(sound, 75, 1000, "Hertz")
    test = []
    test.append(features)
    test = np.asarray(test)
    test = sc.transform(test)
    test1 = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    print(test1.shape)
    predict = cnn.predict(test1)
    predict = np.argmax(predict, axis=1)
    predict = predict[0]
    print(predict)
    label = ['Healthy', 'Parkinson Disease']
    output.insert(END,"Uploaded Speech File : "+filename+"\n\n")
    output.insert(END,"Extracted Features : "+str(test)+"\n\n")
    output.insert(END,"Speech File Predicted as : "+str(label[predict])+"\n\n")

def graph():
    global accuracy, precision, recall, fscore
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['XGBoost','Precision',precision[1]],['XGBoost','Recall',recall[1]],['XGBoost','F1 Score',fscore[1]],['XGBoost','Accuracy',accuracy[1]],
                       ['MLP','Precision',precision[2]],['MLP','Recall',recall[2]],['MLP','F1 Score',fscore[2]],['MLP','Accuracy',accuracy[2]],
                       ['CNN','Precision',precision[3]],['CNN','Recall',recall[3]],['CNN','F1 Score',fscore[3]],['CNN','Accuracy',accuracy[3]],                                                                                    
                      ],columns=['Algorithms','Metrics','Value'])
    df.pivot("Algorithms", "Metrics", "Value").plot(kind='bar')
    plt.show()

font = ('times', 16, 'bold')
title = Label(gui, text='Parkinson Disease Detection Using Deep Neural Networks')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(gui, text="Upload Parkinson Speech Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(gui)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=100)

processButton = Button(gui, text="Preprocess Dataset", command=preprocessDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

svmButton = Button(gui, text="Train SVM Algorithm", command=trainSVM)
svmButton.place(x=270,y=150)
svmButton.config(font=font1)

xgboostButton = Button(gui, text="Train XGBoost Algorithm", command=trainXGBoost)
xgboostButton.place(x=580,y=150)
xgboostButton.config(font=font1)

mlpButton = Button(gui, text="Train MLP Algorithm", command=trainMLP)
mlpButton.place(x=880,y=150)
mlpButton.config(font=font1)

cnnButton = Button(gui, text="Train CNN Algorithm", command=trainCNN)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1)

predictButton = Button(gui, text="Predict Parkinson from Test Speech", command=predictDisease)
predictButton.place(x=270,y=200)
predictButton.config(font=font1)

graphButton = Button(gui, text="Comparison Graph", command=graph)
graphButton.place(x=580,y=200)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
output=Text(gui,height=20,width=150)
scroll=Scrollbar(output)
output.configure(yscrollcommand=scroll.set)
output.place(x=10,y=250)
output.config(font=font1)


gui.config(bg='brown')
gui.mainloop()

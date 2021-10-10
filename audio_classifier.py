# -*- coding: utf-8 -*-
"""
Created on Mon March 21 2020 
@author: Elisheva Guahnich
This audio classifier use an XGBoost a CNN network and the ensemble model
is a logistic regresion between these two models
"""
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, scale, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, UpSampling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from datetime import datetime
#from keras.utils import np_utils
import pandas as pd
import numpy as np
import librosa , csv
import librosa.display
import pylab
import os
import librosa
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import itertools
import scipy as sp
import xgboost as xgb
import joblib
#from imblearn.over_sampling import SMOTE, ADASYN

GOOG = True
if GOOG:
    from google.colab import drive
    drive.mount('/content/drive')
    DIRPATH= '/content/drive/My Drive/' #Google_Colab
else:
    DIRPATH= ''
    

class AudioClassifier():

    def __init__(self):
        self.PLOT_MFCC = True
        self.USE_CNN = True
        self.USE_XGBOOST = False
        self.USE_ENSEMBLE = False
        self.target_names = ['kind_of_sound', 'not_that_kind_of_sound']
        self.dataset = DIRPATH +'audio/cough_dataset.csv'
        self.process_dataset()

    def extract_features(self, file_name):
        try:
            """
                Load and preprocess the audio
            """
            audio, sample_rate = librosa.load(file_name)

            # Remove vocals using foreground separation: https://librosa.github.io/librosa_gallery/auto_examples/plot_vocal_separation.html
            # y_no_vocal = self.vocal_removal(audio, sample_rate)

            # Remove noise using median smoothing
            #y = self.reduce_noise_median(audio, sample_rate)

            # Only use audio above the human vocal range (85-255 Hz)
            # fmin = 260

            # Audio slices
            y = audio
            #fmin = 260
            #fmax = 10000

            """
                Convert to MFCC numpy array
            """
            max_pad_length = 216
            n_mfcc = 60
            n_fft = 2048
            hop_length = 256
            n_mels = 256
            #mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
            mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            pad_width = max_pad_length-mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
            #print(mfccs.shape)
            #mfccsscaled = np.mean(mfccs.T,axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", e)
            return None, None

        return mfccs, sample_rate

    def process_dataset(self):
        features = []
        #Load audio_dataset data 
        audio_dataset_data = DIRPATH + 'audio_dataset/list_data.csv'
        
        #Aditional data
        samples_required = 10000
        not_that_kind_of_sound = 10000
        with open('list_data.csv') as dataset:
            csv_reader = csv.reader(dataset, delimiter=',')
            index = 1
    
            for row in csv_reader:
                file_properties = row[0]
                file_name =  DIRPATH + 'audio_dataset/'+file_properties  #os.getcwd()+'/drive/My Drive/audio_dataset/'+file_properties 
                if row[2] == "kind_of_sound"":
                  class_label = "kind_of_sound"
                else:
                    class_label = "not_that_kind_of_sound"
                try:
                    data, sr = self.extract_features(file_name+ ".wav")
                except:
                    print ("Error extracting features:",file_name+".wav" )
                #print(data)
                if data is not None:
                    if samples_required>=0  or not_that_kind_of_sound>=0:
                        features.append([data, class_label]) 
                        if class_label == "kind_of_sound":
                          samples_required = samples_required - 1
                        else:
                          not_that_kind_of_sound = not_that_kind_of_sound -1
                        # Save an image of the MFCC
                        if self.PLOT_MFCC:
                            self.plot_mfcc(file_properties+'_'+class_label, data, sr)
                        
                else:
                    print("Data is empty: ", file_name)
                #print("audio_dataset Processed row ", index)
                index = index+1

        # Convert into a Panda dataframe
        featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
        print(featuresdf)

        print('Finished feature extraction from ', len(featuresdf), ' files')
        print("\n\nClasses values counts:****************************************")
        print(featuresdf['class_label'].value_counts())
        
        # Convert features and corresponding classification labels into numpy arrays
        X = np.array(featuresdf.feature.tolist())
        y = np.array(featuresdf.class_label.tolist())


        # Encode the classification labels
        
        le = LabelEncoder()
        yy = to_categorical(le.fit_transform(y), num_classes=2)
        if len(featuresdf[featuresdf['class_label']=="not_that_kind_of_sound"])== 0:
            yy[:,0] = 0
            yy[:,1] = 1

        # Train the model and save the results
        if self.USE_CNN:
          self.evaluate_CNN(X, yy)
        if self.USE_XGBOOST:
          self.XGBoost(X, yy, y)
        if self.USE_ENSEMBLE:
          self.ensemble(X, y)

    def denoising_auto_encoder(self, x_train, x_test, num_rows, num_columns, num_channels):
        input_img = Input(shape=(num_rows, num_columns, num_channels))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        #x = tf.placeholder(tf.float32,shape=(None,60,432,32))
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        noise_factor = 0.5
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        
        #batch_size=128
        autoencoder.fit(x_train_noisy, x_train, epochs=30, batch_size=128, shuffle=True, validation_data=(x_test_noisy, x_test), verbose=2)
        x_train = autoencoder.predict(x_train)
        x_test = autoencoder.predict(x_test)
        return x_train, x_test

    def XGBoost(self, X, yy, y):
        # Reshape from 3D to 2D
        X = X.reshape(X.shape[0], -1)
        print(X.shape)

        # PCA dimenstionality reduction
        # pca = PCA(n_components=2)
        # principalComponents = pca.fit_transform(X)
        # principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
        # X = np.array(principalDf['pc1'].tolist())
        # X = pca.components_

        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=42, stratify=y)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        dtrain = xgb.DMatrix(data=x_train, label=y_train)
        dtest = xgb.DMatrix(data=x_test, label=y_test)
        eval_list = [(dtest, 'eval')]

        # Train the model
        params = {
            'max_depth': 3,
            'objective': 'multi:softmax',  # error evaluation for multiclass training
            'num_class': 3,
            'tree_method':'gpu_hist'
        }
        model = xgb.train(params, dtrain, evals=eval_list, early_stopping_rounds=20, verbose_eval=True)

        # Evaluate predictions
        y_pred = model.predict(dtest)
        predictions = [round(value) for value in y_pred]

        accuracy = accuracy_score(y_test.argmax(axis=1), predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # Plots
        #xgb.plot_tree(model)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(150, 150)
        fig.savefig(DIRPATH+ 'xgboost/tree.png')

        # Confusion matrix
        #cm = confusion_matrix(y_train.argmax(axis=1), y_pred.argmax(axis=0))
        #self.plot_confusion_matrix(cm, self.target_names)

        # Save the model
        model.save_model(DIRPATH+ 'saved_models/xgboost_audio_classifier.hdf5')
        print('XGBoost Complete.')

    def ensemble(self, X, y):
        # Encode the classification labels
        le = LabelEncoder()
        yy = to_categorical(le.fit_transform(y))

        # Split into train and test
        num_rows = 60
        num_columns = 432
        num_channels = 1

        x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.5, random_state=42)
        x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
        x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

        # load all models
        members = self.load_all_models()
        print('Loaded %d models' % len(members))

        # evaluate standalone models on test dataset
        for model in members:
            _, acc = model.evaluate(x_test, y_test, verbose=0)
            print('Model Accuracy: %.3f' % acc)

        # fit stacked model using the ensemble
        model = self.fit_stacked_model(members, x_test, y_test)
        # evaluate model on test set
        yhat = self.stacked_prediction(members, model, x_test)
        acc = accuracy_score(y_test.argmax(axis=1), yhat)
        print('Stacked Test Accuracy: %.3f' % acc)

        # Save the model
        filename = DIRPATH + 'saved_models/cnn_ensemble.model'
        joblib.dump(model, filename)
        print('Ensemble Complete.')

    def plot_graphs(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        #plt.show()
        plt.savefig(DIRPATH+'plots/accuracy.png')
        plt.clf()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        #plt.show()
        plt.savefig(DIRPATH+'plots/loss.png')
        plt.close()

    def plot_classification_report(self, x_test, y_test):
        # Print
        print(classification_report(x_test, y_test, target_names=self.target_names))
        # Save data
        clsf_report = pd.DataFrame(classification_report(y_true = x_test, y_pred = y_test, output_dict=True, target_names=self.target_names)).transpose()
        clsf_report.to_csv(DIRPATH+'plots/classification_report.csv', index= True)

    def plot_confusion_matrix(self, cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
        matplotlib.rcParams.update({'font.size': 22})
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(14, 12))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig(DIRPATH+'plots/confusion_matrix.png', bbox_inches = "tight")
        plt.close()

    def plot_mfcc(self, filename, mfcc, sr):
        plt.figure(figsize=(10, 4))
        #S_dB = librosa.power_to_db(mfcc, ref=np.max)
        #librosa.display.specshow(S_dB, y_axis='mel', x_axis='time')
        librosa.display.specshow(librosa.amplitude_to_db(mfcc, ref=np.max), y_axis='mel', x_axis='time', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title(filename)
        plt.tight_layout()
        pylab.savefig(DIRPATH + 'mfcc/'+filename+'.png', bbox_inches=None, pad_inches=0)
        pylab.close()

    def reduce_noise_median(self, y, sr):
        """
            NOISE REDUCTION USING MEDIAN:
            receives an audio matrix,
            returns the matrix after gain reduction on noise
            https://github.com/dodiku/noise_reduction/blob/master/noise.py
        """
        y = sp.signal.medfilt(y,3)
        return (y)

    def vocal_removal(self, y, sr):
        """
            https://librosa.github.io/librosa_gallery/auto_examples/plot_vocal_separation.html
        """
        idx = slice(*librosa.time_to_frames([0, 10], sr=sr))
        S_full, phase = librosa.magphase(librosa.stft(y))
        S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=sr)))
        S_filter = np.minimum(S_full, S_filter)
        margin_i, margin_v = 2, 10
        power = 2
        mask_i = librosa.util.softmask(S_filter,
                                       margin_i * (S_full - S_filter),
                                       power=power)

        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)

        S_foreground = mask_v * S_full
        S_background = mask_i * S_full

        # Convert back to audio
        audio_minus_vocals = librosa.core.istft(S_background[:, idx])

        return audio_minus_vocals

    # load models from file
    def load_all_models(self):
        all_models = list()
        model = load_model( DIRPATH +'saved_models/weights.best.basic_cnn.hdf5')  
        all_models.append(model)
        model = load_model( DIRPATH +'saved_models/xgboost_audio_classifier.hdf5') 
        all_models.append(model)
        return all_models
    	# models = os.listdir(DIRPATH+'saved_models')
    	# for item in models:
    	# 	# define filename for this ensemble
    	# 	filename = DIRPATH+ 'saved_models/'+item
    	# 	#Load model from file
    	# 	model = load_model(filename)
    	# 	#add to list of members
    	# 	all_models.append(model)
    	# 	print('>loaded %s' % filename)

    	# return all_models

    # create stacked model input dataset as outputs from the ensemble
    def stacked_dataset(self, members, inputX):
    	stackX = None
    	for model in members:
    		# make prediction
    		yhat = model.predict(inputX, verbose=0)
    		# stack predictions into [rows, members, probabilities]
    		if stackX is None:
    			stackX = yhat
    		else:
    			stackX = np.dstack((stackX, yhat))
    	# flatten predictions to [rows, members x probabilities]
    	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    	return stackX

    # fit a model based on the outputs from the ensemble members
    def fit_stacked_model(self, members, inputX, inputy):
    	# create dataset using ensemble
    	stackedX = self.stacked_dataset(members, inputX)
    	# fit standalone model
    	model = LogisticRegression(multi_class='ovr')
    	model.fit(stackedX, inputy.argmax(axis=1))
    	return model

    # make a prediction with the stacked model
    def stacked_prediction(self, members, model, inputX):
    	# create dataset using ensemble
    	stackedX = self.stacked_dataset(members, inputX)
    	# make a prediction
    	yhat = model.predict(stackedX)
    	return yhat
    
    def evaluate_CNN(self, X, yy):
        ### CNN
        # Reshape the data
        num_rows = 60 #for CNN only
        #num_rows = 12
        num_columns = 432 #for CNN only
        #num_columns = 60
        num_channels = 1
        num_labels = yy.shape[1]

        y_test = yy
        x_test = X.reshape(X.shape[0], num_rows, num_columns, num_channels)
        print( x_test.shape,  y_test.shape)

        print("kind_of_sound test set:", np.count_nonzero(y_test.argmax(axis=1) == 0))
        print("not_that_kind_of_sound test set:", np.count_nonzero(y_test.argmax(axis=1) == 1))
        # Calculate pre-training accuracy
      
        model = load_model( DIRPATH +'saved_models/weights.best.basic_cnn.hdf5')                
        score = model.evaluate(x_test, y_test, verbose=0)
        y_pred_test = model.predict(x_test, batch_size=15)
        print("Testing Accuracy: ", score[1])
        print ("Confusion matrix test data")
        cm = confusion_matrix(y_test.argmax(axis=1), y_pred_test.argmax(axis=1))
        self.plot_confusion_matrix(cm, self.target_names)
        self.plot_classification_report(y_test.argmax(axis=1), y_pred_test.argmax(axis=1))
        print ('ROC Curve Test data')
        fpr, tpr, _ = roc_curve(y_test, y_pred_test)
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = interp(base_fpr, fpr, tpr)
        print(fpr, tpr)
        
        print('y_pred', y_pred_test )

AudioClassifier()

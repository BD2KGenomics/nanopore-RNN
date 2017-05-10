#!/usr/bin/env python
# http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

from __future__ import print_function
import sys
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from nanonet.features import *
from nanonet.util import all_nmers
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from timeit import default_timer as timer


import os
import re
import subprocess

if "CUDA_HOME" in os.environ:
    utilization = re.findall(r"Utilization.*?Gpu.*?(\d+).*?Memory.*?(\d+)",
                             subprocess.check_output(["nvidia-smi", "-q"]),
                             flags=re.MULTILINE | re.DOTALL)
    print("GPU Utilization", utilization)

    if ('0', '0') in utilization:
        print("Using GPU Device:", utilization.index(('0', '0')))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(utilization.index(('0', '0')))
        os.environ["CUDA_DEVICE_ORDER"]  = "PCI_BUS_ID"  # To ensure the index matches
    else:
        print("All GPUs in Use")
        exit
else:
    print("Running using CPU, NOT GPU")

def grab_s3_files(bucket_path, s3bool=False):
    """Grab the paths to all fast5 files in a s3 bucket or in a local directory"""
    if s3bool:
        bucket = bucket_path.split("/")
        c = boto.connect_s3()
        test = c.lookup(bucket[0])
        if test is None:
            print("There is no bucket with this name!", file=sys.stderr)
            return 1
        else:
            b = c.get_bucket(bucket[0])
        file_paths = []
        for key in b.list("/".join(bucket[1:])):
            if key.name[-5:] == "fast5":
                file_paths.append(os.path.join("s3://", bucket[0], key.name))
        return file_paths
    else:
        onlyfiles = [os.path.join(os.path.abspath(bucket_path), f) for f in os.listdir(bucket_path) if os.path.isfile(os.path.join(os.path.abspath(bucket_path), f)) \
        if f[-5:] == "fast5"]
        # print(onlyfiles)
        return onlyfiles

def get_features(fast5_file, kmer_length=5, window=[-1,0,1], events=False):
    if events == False:
        return events_to_features(get_events_ont_mapping(fast5_file, kmer_len=kmer_length), window=window)
    else:
        return get_events_ont_mapping(fast5_file, kmer_len=kmer_length)

def number_kmers(labels, alphabet='ACGT', length=5, rev_map=False):
    kmers = all_nmers(length, alpha=alphabet)
    bad_kmer = 'X'*length
    kmers.append(bad_kmer)
    all_kmers = {k:i for i,k in enumerate(kmers)}
    return np.fromiter((all_kmers[k] for k in labels),
        dtype=np.int16, count=len(labels))

def get_labels(fast5_file, kmer_length=5):
    labels = get_labels_ont_mapping(fast5_file, kmer_len=kmer_length)
    return number_kmers(labels)

def create_training_input(fast5_files):
    features = get_features(fast5_files[0])
    labels = get_labels(fast5_files[0])
    # features = get_features(fast5_files[0])
    # labels = get_labels(fast5_files[0])
    for fast5 in fast5_files[1:]:
        np.concatenate((features, get_features(fast5)))
        np.concatenate((labels, get_labels(fast5)))
    # print(features[0])
    # print(labels[0])
    return features.reshape(features.shape[0], 1, features.shape[1]), labels



def simpleLSTM(X_train, y_train, X_test, y_test):
    """This works"""
    # fix random seed for reproducibility
    np.random.seed(7)
    print(X_train)
    print(y_train)
    # truncate and pad input sequences
    embedding_vecor_length = 12
    model = Sequential()
    model.add(LSTM(10, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
    model.add(Dense(1025, activation='softmax'))
    # model.add(Dense(1, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam')

    # print(y_train[0])
    y_train = to_categorical(y_train, num_classes=1025)
    # print(y_train[0])
    y_test = to_categorical(y_test, num_classes=1025)

    print(model.summary())
    for i in range(3):
        # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=1, shuffle=False)
        model.fit(X_train, y_train, epochs=1, batch_size=1, shuffle=False)
        model.reset_states()

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0, batch_size=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return model

def simpleLSTM(X_train, y_train, X_test, y_test):
    # fix random seed for reproducibility
    np.random.seed(7)
    print(X_train)
    print(y_train)
    # truncate and pad input sequences
    embedding_vecor_length = 12
    model = Sequential()
    model.add(LSTM(10, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
    model.add(Dense(1025, activation='softmax'))
    # model.add(Dense(1, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam')

    # print(y_train[0])
    y_train = to_categorical(y_train, num_classes=1025)
    # print(y_train[0])
    y_test = to_categorical(y_test, num_classes=1025)

    print(model.summary())
    for i in range(3):
        # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=1, shuffle=False)
        model.fit(X_train, y_train, epochs=1, batch_size=1, shuffle=False)
        model.reset_states()

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0, batch_size=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return model


def train_example():
    # fix random seed for reproducibility
    np.random.seed(7)
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
    print(X_train)
    print(y_train)
    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def scale(train, test):
    """http://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/"""
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

def main():
    start = timer()

    training = "/Users/andrewbailey/personal_git/labWork/davidHaussler/embed_signalalign/two_labeled_files/"

    testing = "/Users/andrewbailey/personal_git/labWork/davidHaussler/embed_signalalign/two_labeled_files/"

    training_files = grab_s3_files(training)
    testing_files = grab_s3_files(testing)
    train_features, train_labels = create_training_input(training_files)
    test_features, test_labels = create_training_input(testing_files)
    # print(train_features[0], train_labels[0])
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler = scaler.fit(train_features)
    # train = train_features.reshape(train_features.shape[0], train_features.shape[1])
    # train_scaled = scaler.transform(train)
    # print(train_scaled[0])
    # train_features = train_features.reshape(train_features.shape[0], 1, train_features.shape[1])
    # test_features = test_features.reshape(test_features.shape[0], 1, test_features.shape[1])

    simpleLSTM(train_features[:100], train_labels[:100], test_features[:100], test_labels[:100])

    # print(features)
    # print(labels)


    stop = timer()
    print("Running Time = {} seconds".format(stop-start), file=sys.stderr)



if __name__=="__main__":
    main()
    raise SystemExit

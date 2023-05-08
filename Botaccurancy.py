import json
import tflearn
import numpy as np

from nltk.stem.lancaster import LancasterStemmer
import pickle
from sklearn.metrics import f1_score, accuracy_score
import nltk
stemmer = LancasterStemmer()

with open("data.json") as file:
    data = json.load(file)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    # data preprocessing
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)
    training = np.array(training)
    output = np.array(output)
    
train_x = training[:int(len(training)*0.8)]
train_y = output[:int(len(output)*0.8)]
test_x = training[int(len(training)*0.8):]
test_y = output[int(len(output)*0.8):]

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8, activation='relu')
net = tflearn.fully_connected(net, 8, activation='relu')
net = tflearn.fully_connected(net, len(train_y[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
    model.save("model.tflearn")   


#eul

predictions = model.predict(test_x)

predicted= np.argmax(predictions, axis=1)
Original = np.argmax(test_y, axis=1)

print("Data Split: 80-20")
f1 = f1_score(Original, predicted, average='weighted')
print('F1-score: ', f1)

num_correct = 0
num_total = len(test_x)
for i in range(num_total):
    if predicted[i] == Original[i]:
        num_correct += 1

error_rate = 1 - (num_correct / num_total)
print("Error rate: ", error_rate)


#acc = confusion_matrix(Original, predicted)
#print('Confusion matrix: ' , acc)


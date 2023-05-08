import nltk
import tflearn
import numpy
import json
from nltk.stem.lancaster import LancasterStemmer
import pickle
stemmer = LancasterStemmer()
# open json file
with open("Test.json") as file:
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
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
# build chatbot model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8, activation='relu')
net = tflearn.fully_connected(net, 8, activation='relu')
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
    model.save("model.tflearn")   
# bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)
userarry=[]
userDenyWords = {"no": "1","I don't feel any": "2","no": "3","that's all": "4"}
#bot response

def botResponse(user_query):
    results = model.predict([bag_of_words(user_query, words)])[0]
    results_index = numpy.argmax(results) 
    tag = labels[results_index]
    if results[results_index] > 0.7:
        for tg in data['intents']:   
            if tg['tag'] == tag:
                responses = tg['responses']
        print("BOT: ", responses)
    else:
        print("BOT: Our system doesn't have the information for this disease")
    userarry.clear()
# chatting
def chat():
    startkeyword=""
    print("Start talking with chatbot(!type start to stop chatting and !type quit to stop chatting)")
    while True:
        user_input = input("YOU: ")
        if user_input.lower() == "quit":
            print("BOT: Thank you for using Health-care bot")
            break
        if startkeyword == "":
            if user_input.lower() != "start":
                print("BOT: Please !type start to start chatting with bot")

            if user_input.lower() == "start":
                startkeyword = "start"
                print("BOT: Can you put sympotoms that you feeling right now?")
        else: 
            userarry.append(user_input)

            if len(userarry) != 3:
                print("BOT: Any information you can provide us")
                if user_input in userDenyWords:
                    print("BOT: Can't show results , need more symtoms!")
                    userarry.clear()
            if len(userarry) == 3:
                query = ",".join(userarry)
                botResponse(query)   
chat()

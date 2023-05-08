import tkinter as tk
import nltk
import tflearn
import numpy
import json
from nltk.stem.lancaster import LancasterStemmer
import pickle


stemmer = LancasterStemmer()


# open json file
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
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# build chatbot model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    # train chatbot model
    model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

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
        return responses[0]  
    else:
        return "Our system doesn't have the information for this disease"

userarry.clear()
userarry=[]
userDenyWords = {"no": "1","I don't feel any": "2","no": "3","that's all": "4"}

def send():
    user_input = input_box.get()
    input_box.delete(0, tk.END)
    if not user_input:
        chat_window.insert(tk.END, "BOT: Welcome! To start the chat, please type 'start'.\n")
        return
    
    chat_window.insert(tk.END, "You: " + user_input + "\n\n")
    if user_input.lower() == "quit":
        chat_window.insert(tk.END, "BOT: Thank you for using Health-care bot\n")
        root.destroy()
        return
    if user_input.lower() == "start":
        chat_window.insert(tk.END, "BOT: Can you put symptoms that you're feeling right now?\n")
    else:
        userarry.append(user_input)
        if len(userarry) != 3:
            chat_window.insert(tk.END, "BOT: Any information you can provide us\n")
            if user_input in userDenyWords:
                chat_window.insert(tk.END, "BOT: Can't show results, need more symptoms!\n")
                userarry.clear()
            return
        if len(userarry) == 3:
                query = ",".join(userarry)
                botResponse(query) 
        result = botResponse(query)
        chat_window.insert(tk.END, "BOT: "+"'" + result +"'"+"\n")
        userarry.clear()
root = tk.Tk()
root.title("Chatbot")
root.geometry("400x500")

root.configure(bg="#212121")
chat_window = tk.Text(root, bd=0, bg="#5D6D7E", height="8", width="50", font="Arial")
scrollbar = tk.Scrollbar(root, command=chat_window.yview, cursor="heart")
chat_window['yscrollcommand'] = scrollbar.set
input_box = tk.Entry(root, bd=0, bg="#2C2F33", width="30", font="Arial", fg="white")
send_button = tk.Button(root, text="Send", command=send, bg="#4286f4", activebackground="#3c9dc8", fg='#ffffff', bd=0, highlightthickness=0, font="Arial")

scrollbar.place(x=376,y=6, height=386)
chat_window.place(x=15,y=6, height=386, width=361)
input_box.place(x=128, y=401, height=30, width=265)
send_button.place(x=15, y=401, height=30)

root.mainloop()
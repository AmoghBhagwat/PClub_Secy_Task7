from flask import Flask, request, render_template
import torch
from dataset import Dataset
from model import Model
import random
import numpy as np

app = Flask(__name__)

dataset = Dataset(4)
model = Model(dataset)
model.load_state_dict(torch.load('model1.pt'))
model.eval()

def generate(next_words=5):
    text = dataset.words[random.randint(0,len(dataset.get_unique_words())-1)]
    words = text.split(' ')
    model.eval()

    hidden, cell = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (hidden, cell) = model(x, (hidden, cell))

        previous_word = y_pred[0][-1]
        p = torch.nn.functional.softmax(previous_word, dim=0).detach().numpy()
        word_index = np.random.choice(len(previous_word), p=p)
        words.append(dataset.index_to_word[word_index])

    return words

@app.route('/')
def main():
    return render_template('index.html', clickbait='')

@app.route('/', methods=['POST'])
def generate_clickbait():
    if request.method == 'POST':
        words = generate(next_words=6)
        sentence = ''
        for word in words:
            sentence += word
            sentence += ' '
        return render_template('index.html', clickbait=str(sentence).upper())
    

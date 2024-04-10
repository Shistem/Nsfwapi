from flask import Flask, request, jsonify
import json
import tensorflow as tf
import numpy as np
import random
import pickle
import re

app = Flask(__name__)

# Load the tokenizer and model
with open('nsfw_classifier_tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

with open('nsfw_classifier.pickle', 'rb') as f:
    model = pickle.load(f)

# Define the vocabulary size and embedding dimensions
vocab_size = 10000
embedding_dim = 64

# Pad the prompt and negative prompt sequences
max_sequence_length = 50

def preprocess(text, isfirst=True):
    if isfirst:
        if type(text) == str:
            pass
        elif type(text) == list:
            output = []
            for i in text:
                output.append(preprocess(i))
            return output

    text = re.sub('<.*?>', '', text)
    text = re.sub('$$+', '(', text)
    text = re.sub('$$+', ')', text)
    matches = re.findall('$$.*?$$', text)

    for match in matches:
        text = text.replace(match, preprocess(match[1:-1], isfirst=False))

    text = text.replace('\n', ',').replace('|', ',')

    if isfirst:
        output = text.split(',')
        output = list(map(lambda x: x.strip(), output))
        output = [x for x in output if x != '']
        return ', '.join(output)

    return text

def postprocess(prompts, negative_prompts, outputs, print_percentage=True):
    for idx, i in enumerate(prompts):
        print('*****************************************************************')
        if print_percentage:
            print(f"prompt: {i}\nnegative_prompt: {negative_prompts[idx]}\npredict: {outputs[idx][0]} --{outputs[idx][1]}%")
        else:
            print(f"prompt: {i}\nnegative_prompt: {negative_prompts[idx]}\npredict: {outputs[idx][0]}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prompt = data['prompt']
    negative_prompt = data['negative_prompt']

    x_new = tokenizer.texts_to_sequences(preprocess(prompt))
    z_new = tokenizer.texts_to_sequences(preprocess(negative_prompt))
    x_new = tf.keras.preprocessing.sequence.pad_sequences(x_new, maxlen=max_sequence_length)
    z_new = tf.keras.preprocessing.sequence.pad_sequences(z_new, maxlen=max_sequence_length)
    y_new = model.predict([x_new, z_new])
    y_new = list(map(lambda x: ("NSFW", float("{:.2f}".format(x[0] * 100))) if x[0] > 0.5 else ("SFW", float("{:.2f}".format(100 - x[0] * 100))), y_new))

    response = {
        'prediction': y_new
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()

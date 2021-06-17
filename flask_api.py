#load imortant modules
import os
import collections
import pathlib
import re
import json
import string
import pandas as pd
import numpy as np
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from flask import Flask,request

from official.modeling import tf_utils
#from flask_ngrok import run_with_ngrok
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tensorflow_datasets as tfds
import flasgger
from flasgger import Swagger

# data, info = tfds.load('glue/mrpc', with_info=True,
#                        # It's small, load the whole dataset
#                        batch_size=-1)

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)

#flask app
app = Flask(__name__)
#run_with_ngrok(app)
Swagger(app)

# gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
# tf.io.gfile.listdir(gs_folder_bert)

model = tf.saved_model.load('model')

#flask app
@app.route('/')
def Hello_world():
  return "<h1>Hello World!</h1>"

@app.route('/predict',methods=["Get"])
def get_prediction():

  """Lets compare sentences and see if they convey the same menaing!
  ---
  parameters:
    - name: sentence_1
      in: query
      type: string
      required: true
    - name: sentence_2
      in: query
      type: string
      required: true
  responses:
    200:
      description: The predicted value
  """
  
  tokenizer = bert.tokenization.FullTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
     do_lower_case=True)

  def encode_sentence(s, tokenizer):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)

  def bert_encode(glue_dict, tokenizer):
    num_examples = len(glue_dict["sentence1"])

    sentence1 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence1"])])
    sentence2 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence2"])])

    cls_ = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
    input_word_ids = tf.concat([cls_, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls_)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs

  sentence_1 = request.args.get('sentence_1')
  senetnce_2 = request.args.get('sentence_2')


  example = {'sentence1':[sentence_1],'sentence2':[senetnce_2]}


  preprocessed_example = bert_encode(
    glue_dict = example ,
    tokenizer=tokenizer)
  
  proba = model([preprocessed_example['input_word_ids'],
                            preprocessed_example['input_mask'],
                            preprocessed_example['input_type_ids']], training=False)

  
  result = result = tf.argmax(proba[0]).numpy()

  return "Probality score  and result" + " " + "Result-" + " " + str(result) + " " + "Probability" + " " + str(proba)


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)

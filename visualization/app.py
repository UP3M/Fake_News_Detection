import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from keras.utils import pad_sequences
import pandas as pd
import torch
import tensorflow as tf
from keras.preprocessing.text import tokenizer_from_json

@st.cache_resource
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # change the path of fine-tuned BERT model here:
    model = BertForSequenceClassification.from_pretrained("C:/Users/oka resia/Downloads/NLP/Fake news detection/pretrained model/fakenews_BERT")

    # change the path of fine-tuned 1D-CNN model here:
    new_model = tf.keras.models.load_model("C:/Users/oka resia/Downloads/NLP/Fake news detection/pretrained model/fakenews_CNN/cnn_pretrain")
    return tokenizer,model,new_model


tokenizer,model,cnn_model = get_model()

st.write("""
# Fake News Detection !
""")

headline_input = st.text_area('Enter Headline to Analyze')
body_input = st.text_area('Enter Article Body to Analyze')

button_bert = st.button("Analyze with **BERT**")
button_tfcnn = st.button("Analyze with **Custom Model**")

ids_to_labels = {0:"unrelated", 1:"discuss", 2:"disagree", 3:"agree"}

if headline_input and body_input and button_bert :
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    data = {'articleBody':[body_input], 'Headline':[headline_input]}
    user_input = pd.DataFrame(data)
    test_sample = tokenizer(user_input.values.tolist(), padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    input_id = torch.tensor(test_sample['input_ids']).to(device)
    mask = torch.tensor(test_sample['attention_mask']).to(device)

    output = model(input_id, mask, None)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediction: ",ids_to_labels[y_pred[0]])

if headline_input and body_input and button_tfcnn :
    MAX_NB_WORDS = 50000
    MAX_SEQUENCE_LENGTH = 250

    # Load the tokenizer from file
    # change the path of pretrained JSON for headline here:
    with open('C:/Users/oka resia/Downloads/NLP/Fake news detection/pretrained model/fakenews_CNN/cnn_tokenizer/tokenizer_head.json', 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)

    # Convert the new data to sequences of integers
    new_headline_seq = tokenizer.texts_to_sequences([headline_input])[0]
    # Pad the sequences to have the same length
    new_headline_seq = pad_sequences([new_headline_seq], maxlen=MAX_SEQUENCE_LENGTH)

    # change the path of pretrained JSON for body of the article here:
    with open('C:/Users/oka resia/Downloads/NLP/Fake news detection/pretrained model/fakenews_CNN/cnn_tokenizer/tokenizer_body.json', 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
    new_article_seq = tokenizer.texts_to_sequences([body_input])[0]
    new_article_seq = pad_sequences([new_article_seq], maxlen=MAX_SEQUENCE_LENGTH)

    # Preprocess the new data (assume new_headline and new_article are already preprocessed)
    new_data = [new_headline_seq, new_article_seq]
    # Make predictions
    y_pred = cnn_model.predict(new_data)

    # Get the predicted class
    predicted_class = np.argmax(y_pred)

    # Get the predicted probability for each class
    predicted_probabilities = y_pred[0]

    # Print the predicted class and probability for each class
    st.write("Prediction: ",ids_to_labels[predicted_class])
    st.write("Predicted probabilities: ",y_pred)
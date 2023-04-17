# Fake News Detection


## 1. Introduction

Fake news (Hoax) on the internet has become a global problem that causes turmoil in society. Its presence can disrupt democratic order, the stability of social, cultural, political and economic life. The existence of fake news can disrupt democratic order, the stability of social, cultural, political and economic life. There have been many companies whose shares have fallen due to the presence of fake news among the public about these companies. In addition, for business people, if they don't filter fake news, they will experience failure in taking business steps, one of which is to misjudge the market. In fact, more than half of online news stories from its global sample (55%) express strong concern or concern about “what is real or fake.” This amount varies greatly in different countries. For example, at the top of the list are Brazil (85%), South Africa (70%), Mexico (68%) and France (67%).

In addition to the information provided through public reports, a mechanism is needed to control and reduce the spread of fake news. Lately there have been many studies and studies on this matter, even adopting the epidemic principle to map the pattern of spreading fake news on social networks. It was found that the processes of epidemic transmission and information dissemination have the same theoretical pattern. One of the techniques to control and reduce fake news is to create a system that can classify news automatically.


## 2. Existing Approaches

Fake news detection is an important task in natural language processing (NLP) and has received considerable attention in recent years. Various approaches have been proposed to tackle this problem, some of which are:

### 2.1. Supervised Machine Learning. 

This approach involves training a machine learning model on a dataset of labeled examples. The model is then used to classify news articles as either real or fake based on various features such as the text content, metadata, and source information.

### 2.2. Natural Language Processing Techniques.

These techniques involve analyzing the language used in news articles to detect patterns and anomalies that may indicate fake news. This can include techniques such as sentiment analysis, part-of-speech tagging, and topic modeling.

### 2.3. Graph-based Approaches.
 
These approaches involve analyzing the relationships between different news articles and sources to identify patterns that may indicate fake news. This can include techniques such as network analysis, link analysis, and clustering.

### 2.4. Deep Learning

Deep learning techniques, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs), have also been used to detect fake news. These techniques involve training a neural network on a dataset of news articles to learn patterns that indicate whether an article is real or fake.

### 2.5. Ensemble Methods.
 
This approach involves combining multiple models or techniques to improve the accuracy of fake news detection. For example, a combination of supervised learning, natural language processing, and graph-based approaches can be used to build a more robust fake news detection system.


## 3. Implementation of Fake News Detection

With current NLP, labeling the truth is practically difficult. In fact, it's difficult for even humans to tell the difference between true and false news. The data that does exist (e.g., fact checker website archives) is almost entirely copyright protected, the data that does exist is extremely diverse and unstructured, making it difficult to train on, and any dataset containing claims with associated "truth" labels is going to be contested as biased. These factors all contribute to the lack of labeled training data of fake vs. real news stories. Therefore, we concluded that developing a strong stance detection system would enable a human fact-checker to enter a claim or headline and rapidly return the best articles that support/agree, refute/disagree, or discuss the statement in question. After examining the reasons in favor of and against the claim, they might evaluate its veracity using human judgment and reasoning. Human fact-checkers might be made quick and efficient with the help of such a tool.

### 3.1. Implementation of Fake News Detection using TF-IDF

TF-IDF is a useful method for detecting fake news, where each news article can be converted into a vector of numerical features that highlight the significant words and phrases. Such a vector can then be used to train a machine learning algorithm to classify fake news. By comparing the TF-IDF vectors of multiple news articles, we can spot the words and phrases that are most typical of fake news and leverage them to train a model that can differentiate between fake news and real news.
TF-IDF is a potent technique for analyzing text data and has proven to be beneficial for fake news detection. However, besides it being the fastest model among 1D-CNN and BERT, the results show that TF-IDF has the worst performance in terms of accuracy. TF-IDF was only able to achieve around 0,26 accuracy score which is very bad. We found that TF-IDF may not be sufficient to identify all instances of fake news, and additional features and techniques may be necessary to enhance the accuracy of the model.

### 3.2. Implementation of Fake News Detection using 1D-CNN

One-dimensional convolutional neural networks (1D CNN) have been increasingly utilized in the detection of fake news. These models are capable of analyzing the textual content of news articles and detecting patterns that indicate the presence of misinformation or disinformation.
The selection of this particular design for the model was based on its ease of use and ability to perform rapid calculations by utilizing convolutions instead of recurrence. The model's efficacy appears to be attributed to the convolution's ability to capture diverse themes, however, it is limited in that it can only process the text once. To improve the model, it could be enhanced by adding an attention mechanism with recurrence after the convolution. This would enable the model to query specific components of the headline or body of the text after receiving a general summary from CNN.
{model description} 1D-CNN itself was able to give 0.8 in terms of accuracy score which is better than TF-IDF in terms of accuracy and also better than BERT in terms of training speed which only needs 30 minutes to finish the training step.

### 3.3. Implementation of Fake News Detection using BERT

BERT is an advanced pre-trained word embedding model based on transformer encoded architecture. We utilize BERT as a sentence encoder, which can accurately get the context representation of a sentence. BERT removes the unidirectional constraint using a mask language model (MLM). It randomly masks some of the tokens from the input and predicts the original vocabulary id of the masked word based only. MLM has increased the capability of BERT to outperforms as compared to previous embedding methods. It is a deeply bidirectional system that is capable of handling the unlabelled text by jointly conditioning on both left and right context in all layers. In this research, we have extracted embeddings for a sentence or a set of words or pooling the sequence of hidden-states for the whole input sequence. A deep bidirectional model is more powerful than a shallow left-to-right and right-to-left model.
{model description} BERT itself was able to give 0.9 in terms of accuracy score which is better than TF-IDF and 1D-CNN (alone) in terms of accuracy. However, in terms of speed, to fine-tune the model using pre-trained BERT, it took around 5 hours to finish the whole process.


## 4. Conclusion

In conclusion, TF-IDF involves converting news articles into a vector of numerical features to train a machine learning algorithm. Although it is fast, it had the worst performance in terms of accuracy, only achieving a score of 0.26. The 1D CNN model uses convolutions to analyze textual content and detect patterns indicating misinformation or disinformation. It achieved an accuracy score of 0.8, which was better than TF-IDF and faster than BERT. BERT, a pre-trained word embedding model, removes the unidirectional constraint using a mask language model, and is capable of handling unlabelled text. It achieved the highest accuracy score of 0.9, but it took around 5 hours to fine-tune the model. Overall, while each method has its strengths and weaknesses, BERT appears to be the most effective method for fake news detection, but may not be practical in terms of time and computational resources.

## How To Run:
#### 1. The dataset is available in "dataset" folder.
#### 2. The pretrained model for 1D-CNN and fine-tuned model for BERT are available on https://drive.google.com/file/d/1D2Dwf15oGeeeQgGcnZTnJ8baNHCILR75/view?usp=sharing.
#### 3. We visualize our project using streamlit (all necessary file in "visualization" folder), to clone it and make it work, simply do following step:
 3.1. Download all pretrained model and change the path inside app.py
 
 3.2. Install all necessary library by using pip command (all required library is writen in requirement.txt)
 
 3.3. Run streamlit using command "streamlit run app.py"
 
 ##### N.B.: recommended to use virtual environment


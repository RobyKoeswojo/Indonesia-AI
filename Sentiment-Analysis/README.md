# 2019 Indonesian Presidential Election Tweet Sentiment Analysis


## Overview

2019 Indonesian Presidential Election is a general election held in Indonesia to elect the president and vice president for the period of 2019 to 2024. The main candidates are the incumbent President Joko Widodo and Ma'ruf Amin against Prabowo Subianto and his running mate Sandiaga Uno.  

A collection of tweet related to both candidates is collected, each of which is labeled as negative, neutral or positive sentiment. The goal is to build a machine learning model which can help classifying the sentiment of each tweet automatically.

In this project, various text cleaning techniques to remove stopwords, slang words, etc are employed. On top of that, different types of word embeddings, namely TF-IDF, Word2Vec, and embedding layer from Keras are used to encode the text, so that they can be fed as the input of the machine learning model.
Two machine learning algortihms are experimented. They are Random Forest and LSTM based network. Benchmarking the best result to the pre-trained models is also done.


## Dataset

The dataset consists of 1815 tweets related to the main candidates with their corresponding sentiment.
The distribution of the sentiments is as following:

| Class      | Frequency (Count) | Proportion |
| ---------- | ----------------- | ---------- |
| Positive   | 612               | 33.72%     |
| Neutral    | 607               | 33.44%     | 
| Negative   | 596               | 32.8%      | 


### Exploratory Data Analysis

**Hashtags**

Counting the hashtags with respect to the sentiment of the texts shows that hashtags in the texts are inconclusive to a specific sentiment

| ![Top 5 hashtags for each sentiment](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/hasthags_exploration.PNG?raw=true) |
|:--:| 
| Top 5 hashtags for each sentiment |


**Candidates Names**  

The candidates names mentioned in each tweet shows that both candidates are balancely mentioned for each sentiment

| ![Candidates names distribution at each sentiment](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/president_names_exploration.PNG?raw=true) |
|:--:| 
| Candidates names distribution at each sentiment |


**WordCloud**

Looking at the WordCloud, it can be seen that some keywords only appear for a candidate, e.g. the keyword 'gaji' appears for Prabowo, whereas 'harga' appears for Jokowi.


| ![Wordcloud for Jokowi with positive sentiment](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/wordcloud_jokowi_positif.png?raw=true) |
|:--:| 
| Wordcloud for Jokowi with positive sentiment |

| ![Wordcloud for Prabowo with positive sentiment](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/wordcloud_prabowo_positif.png?raw=true) |
|:--:| 
| Wordcloud for Prabowo with positive sentiment |


**Visualization**

TF-IDF vectorizer is used to vectorize the texts, and LDA (Latent Dirichlet Allocation) is used to reduce the dimension of the embedded words, so that visualization can be made.
The visualization shows how the embedded words corresponding to each sentiment are spread out evenly -- no specific cluster is made. This may cause the model difficult to classify the sentiments.  

| ![Embedded Words Visualization](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Sentiment-Analysis/images/word_projection.PNG) |
|:--:| 
| Embedded words visualization |


## Text Preprocessing

| ![Text cleaning workflow](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/text_cleaning.PNG) |
|:--:| 
| Text cleaning workflow |


The text cleaning steps are as following:
1. Remove any url appears in the text
2. Remove any hashtag in the text
3. Normalize the slang words in the text (refer to the slang word vocab taken from [here](https://github.com/nasalsabila/kamus-alay/blob/master/colloquial-indonesian-lexicon.csv))
4. Run word postagging to tag each word at nlp-id's stopwords, and take only stopwords that are no negation (NEG), adjective (JJ), verb (VB), foreign words (FW) and numbers (NUM) as the custom stopwords
5. Remove stopwords by referring to the newly-made custom stopwords
6. Run phrase postagging on the preprocessed texts, normalize numbers to *NUM* and lemmatize only the verbs


**Data Splitting**  
20% of the text in the dataset is taken as the validation set, where the remaining texts are used as the train set.  
The sampling strategy is stratified sampling.


**Label Encoding**  
Label encoding is done to convert the labels (positive, neutral, negative) into numbers (2, 1, 0)


## Embedding and Modeling

The word embeddings used are TF-IDF, Word2Vec and Embedding Layer at Keras.
The main models experimented are Random Forest and LSTM-based network.   

The validation accuracy result is as following:
| Class         | TF-IDF | Word2Vec | Embedding Layer Tensorflow |
| ------------- | ------ | -------- | -------------------------- |
| Random Forest | 61.15% |  61.4%   |             -              |
| LSTM          | 57.38% |  58.9%   |          58.41%            |

The best validation accuracy obtained is 61.4%, which uses Random Forest classifier with Word2Vec embedding.


**Findings**  
In general, Random Forest model performs better than LSTM-based network in this use case. 
However, all experiments suffer from overfitting, which can be seen from the learning curve, where it shows a good training performance which is not followed by improvement by the validation performance

   
| ![Learning Curve for LSTM + Embedding Layer](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/lstm_embedding_learning_curve.png?raw=true) |
|:--:| 
| Learning curve for LSTM + embedding layer |


## Benchmarking

A pre-trained BERT model by mdhugol/indonesia-bert-sentiment-classification (https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification) is employed to predict the sentiment on the validation set.

The validation accuracy using the pre-trained BERT model is 63.63%.  
This shows that the performance of the Random Forest with Word2Vec is comparable to the pre-trained BERT model for this use case.

The confusion matrix on the predicted validation set by the BERT and Random Forest classifier show similar behavior, where prediction on the negative class is better (> 70% accuracy), followed by neutral and positive class.

| ![Confusion Matrix for BERT (left) and Random Forest + Word2Vec (right)](https://github.com/RobyKoeswojo/Indonesia-AI/blob/sentiment_analysis/Sentiment-Analysis/images/cm_bert_rfw2v.PNG) |
|:--:| 
| Confusion matrix for BERT (left) and Random Forest + Word2Vec (right) |

## Conclusion
- For this specific use case, the best model is random forest with word2vec word embedding, which produces validation accuracy of 61.7%
- Benchmarking to pretrained models is done, and the best pre-trained model is the BERT model with validation accuracy of 63.63%
- The models still suffer from overfitting

## Improvement
- The text cleaning process still can be improved. Current texts still have slang informal words detected
- Try using other machine learning algorithms (not deep learning algorithms) for this use case, and fix the overfitting issue
- Run error analysis on the predicted results, and focus on fixing from the most misclassified class

This sentiment analysis task is part of the projects done during NLP bootcamp at Indonesia AI.
Other projects during the NLP bootcamp can be found at https://github.com/RobyKoeswojo/Indonesia-AI
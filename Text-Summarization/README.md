# TEXT SUMMARIZATION - LIPUTAN6 ARTICLES SUMMARIZATION


## OVERVIEW
Liputan6 is an Indonesian television news program that broadcasts on Indonesian channels: SCTV and Moji. News articles published on [liputan6.com](!https://www.liputan6.com/) and their corresponding summary are available at [huggingface](https://huggingface.co/datasets/id_liputan6). More details can be seen at the huggingface page.


## SCOPE
### Dataset
The scope for this text summarization task focuses on the canonical variant of the dataset, and because of the computing resource limitation, only 10% of total data are used.  


### Resources
The main resource used for this text summarization task is the free version of google colab
	- 12.7 GB RAM
	- T4 GPU (limited time)


### Preprocessing
The preprocessing step done for this text summarization task is simply removing the *liputan6.com:* word at each article.


### Models
The models explored are pre-trained language models that are available at huggingface: [BERT2GPT-indonesian-summarization](https://huggingface.co/cahya/bert2gpt-indonesian-summarization), [FLAN-T5 XL](https://huggingface.co/google/flan-t5-xl) and [FLAN-T5 Small](https://huggingface.co/google/flan-t5-small)


### Evaluation
The main evaluation metrics is Rouge. The rouge-1, rouge-2, rouge-3 and rouge-L score for each generated summary is calculated and averaged.


## EXPLORATORY DATA ANALYSIS

**Articles and Summaries Length**

| ![Length of words of the articles and summaries at the train set](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Text-Summarization/images/train_length.png) |
|:--:| 
| Length of words of the articles and summaries at the train set |


| ![Length of words of the articles and summaries at the validation set](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Text-Summarization/images/validation_length.png) |
|:--:| 
| Length of words of the articles and summaries at the validation set |


In average, the length of the articles and summaries at each set:
| Length of Words      | Train | Validation |
| -------- | ----- | ---------- |
| Article  | 1168  | 1368       |
| Summary  | 170   | 174        | 


It can be seen that the length of the articles and summaries in both the training and validation set are similar.

**Top 10 N-gram words**

| ![Top 10 unigram words at train set](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Text-Summarization/images/top10_1gram_train.png) |
|:--:| 
| Top 10 unigram words at train set |


| ![Top 10 unigram words at validation set](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Text-Summarization/images/top10_1gram_validation.png) |
|:--:| 
| Top 10 unigram words at validation set |


| ![Wordcloud for train data](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Text-Summarization/images/wordcloud_train.png) |
|:--:| 
| Wordcloud for train data |


| ![Wordcloud for validation data](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Text-Summarization/images/wordcloud_validation.png) |
|:--:| 
| Wordcloud for validation data |


By checking the number of occurences of the unigram words, it can be seen that both in the training and validationd data,
the conjunction words are dominating. Since the summaries also contain conjuction words, the conjunctions words will not be removed from the articles.


| ![Top 10 bigram words at train set](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Text-Summarization/images/top10_2gram_train.png) |
|:--:| 
| Top 10 bigram words at train set |


| ![Top 10 bigram words at validation set](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Text-Summarization/images/top10_2gram_validation.png) |
|:--:| 
| Top 10 bigram words at validation set |


Similarly, the top 10 bigram words also shows that the conjunction words are the most dominating words in both the articles and summaries.



## PREDICTION

### Indonesian BERT2GPT Summarization
This pretrained model is a model that has been fine-tuned on the id-liputan6 dataset (the dataset which will be predicted in this task).
| Architecture       | Encoder (BERT) - Decoder (GPT2) model |
| Fine-tuned dataset | id_liputan6			     |


The average rouge score is as following:
| Rouge-1 | Rogue-2 | Rouge-3 | Rouge-L |
| ------- | ------- | ------- | ------- |
| 0.373   | 0.188   | 0.109   | 0.292   |


Samples of generated summaries:
| ![Samples of generated summaries by Indonesia Bert2bert Summarization](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Text-Summarization/images/generated_sum_bert2bert.PNG) |
|:--:| 
| Samples of generated summaries by Indonesia Bert2bert Summarization |


### FLAN-T5 XL
FLAN-T5 model is a T5 model that is fine-tuned on FLAN collection dataset by Google. The XL variant is already trained to understand Bahasa Indonesia. This model has 2.85 Billion parameters.

| Architecture       | T5                      |
| Fine-tuned dataset | FLAN collection dataset |


The average rouge score is as following:
| Rouge-1 | Rogue-2 | Rouge-3 | Rouge-L |
| ------- | ------- | ------- | ------- |
| 0.255   | 0.128   | 0.072   | 0.197   |


Samples of generated summaries:
| ![Samples of generated summaries by FLAN-T5-XL](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Text-Summarization/images/generated_sum_flant5xl.PNG) |
|:--:| 
| Samples of generated summaries by FLAN-T5-XL |


**Result**
The FLAN-T5-XL model, although understands Bahasa Indonesia, does not perform better than the bert2gpt model, e.g. repeated words are generated by the FLAN-T5-XL model. The underperforming result is possibly because the bert2gpt model has been finetuned on the specific id-liputan6 dataset, whereas the FLAN-T5-XL is finetuned on different dataset.  


### PEFT FLAN-T5 Small
To improve the FLAN-T5 result, the idea is to fine-tune the model with id-liputan6 dataset. However, due to the limitation of the GPU usage by free google colab, the model will not be full fine-tuned. A technique called Parameter Efficient Fine Tuning (PEFT) is employed on a smalled model, i.e. FLAN-T5-Small model.

FLAN-T5 small is the smallest variant of the FLAN-T5 models. This model only has 77 Million parameters and this model does not understand Bahasa Indonesia yet.  

| Architecture       | T5 |
| Fine-tuned dataset | FLAN collection dataset  |


The original idea is to train the FLAN-T5-small model to perform text summarization on the id-liputan6 dataset.
However, due to the computing resource limitation, full fine tuning is not possible. Therefore, in this task, Parameter-Efficient-Fine-Tuning (PEFT) is employed.  

PEFT method has several approaches, e.g. soft prompt, Low Rank Adaptation (LoRA), etc. The focus on this task is LoRA, which is a popular method for parameter efficient fine tuning.  

Recall that in transformer models, vectors to be fed into the self attention layer are decomposed into three main parts, i.e. Q, K, V matrix -- Query, Key, Value matrix, respectively. The LoRA method basically adds a low rank matrix to these Q, K, V matrices. In this experiment, the LoRA uses rank 32 and the additional matrices are added to Q and V modules.  

By running the LoRA method, the number of parameters updated are only 130K parameters, instead of updating the whole 77 million parameters.


The average rouge score is as following:
| Rouge-1 | Rogue-2 | Rouge-3 | Rouge-L |
| ------- | ------- | ------- | ------- |
| 0.062   | 0.023   | 0.011   | 0.055   |


Samples of generated summaries:
| ![Samples of generated summaries by PEFT-FLAN-T5-Small](https://github.com/RobyKoeswojo/Indonesia-AI/blob/main/Text-Summarization/images/generated_sum_peftflant5small.PNG) |
|:--:| 
| Samples of generated summaries by PEFT-FLAN-T5-Small |


**Result**
The summaries generated by the PEFT FLAN-T5-Small model also contain repeating words, and moreover, some words generated are in English. The result is logical since this model does not understand Bahasa Indonesia yet. The underperforming result is also reflected by the low Rouge scores.


## CONCLUSION
1. The best result is using the BERT2GPT-indonesian-summarization, which achieves:

| Rouge-1 | Rogue-2 | Rouge-3 | Rouge-L |
| ------- | ------- | ------- | ------- |
| 0.373   | 0.188   | 0.109   | 0.292   |


2. The main challenge in this task is the computing resource limitation, such that fine tuning process and the usage of larger and more complex model are not possible.

## IMPROVEMENT
If there is no constraint in the cost, using paid LLM such as OpenAI model produces much better results. Renting a higher GPU service is also an alternative, which enables full fine tuning and using more complex model.

This text summarization task is part of the projects done during NLP bootcamp at Indonesia AI.
Other projects during the NLP bootcamp can be found at https://github.com/RobyKoeswojo/Indonesia-AI
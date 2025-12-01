# Potential Talents - Retriever of relevant job titles 

Updated Dec 1st 2025 

## Raw Data 
For each candidate, we are provided with the following features 
* id : unique identifier for candidate (numeric)
* job_title : job title for candidate (text)
* location : geographical location for candidate (text)
* connections: number of connections candidate has, 500+ means over 500 (text)

## Goal(s) 
* Predict how fit the candidate is based on their available information (variable fit) 
* Rank candidates based on a fitness score. 
* Re-rank candidates when a candidate is starred. 

## Tech stack 
### Python packages 
* pandas 
* numpy 
* sklearn (For tf-idf)
* gensim (For training the Word2Vec and the FastText models, and loading the pre-trained GloVe model) 
* sentence_transformer (For loading the pre-trained SBert model) 

## Models 
### General structure
For each of the following models, we construct a python class "\<base model name\>Ranker". 
This Ranker will fit the base model (or load the base model) to the carpus consisted of the "job_title" feature, embed the "job_title" features, and calculate the cos similarity score between the "job_title" feature and the queried title as the score for ranking. To "star" certain job titles, one put the stared titles in as a new query and query again, the ranker will construct a new score as linear combination of the old score and the new cos similarity score to provide an updated ranking. 

### Base models used 
* TF-IDF sentence embedding model 
* Word2Vec (SG-NS, and CBOW) word embedding with mean sentence embedding model 
* Glove word embedding with TF-IDF weighted sentence embedding model 
* FastText word embedding with TF-IDF sentence embedding model 
* SBert sentence embedding model 

For each of above models, a corresponding ipynb notebook can be found in [this linked folder](/note_books/) containing its process and theoretical review. One can also locate the raw code in [this linked folder](/proj_mod/) for each of the above models. 

## Evaluation 
This task is unsupervised in the sense that there is no pre-determined "target" value. 
All evaluation method requires some further outside engagement, and some suggested common method can be seen in [this linked document](/note_books/suggested_metrics.ipynb). 

## Future
For now, the ranking only relies on the "job_title" feature, it might some worth effort to look into utilizing otherwise features (e.g. calculate geographic distance according to "location" feature, and have that impact the results as well). 
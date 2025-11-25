from gensim.models.fasttext import FastText
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Union, List
import pandas as pd 
import numpy as np 

class FastTextRanker: 
    def __init__(self):
        pass 
    
    def fit(self, 
            df: pd.DataFrame,
            fasttext_kwards: Union[dict, None]=None, 
            vector_size:int=300, 
            window: int=5, 
            min_count: int=2, 
            sg: int=1, 
            epochs: int=5, 
            min_n: int=2, 
            max_n: int=6, 
            bucket: int=2000000, 
            workers: int=8): 
        self.df=df.sort_values(by="id")
        self.corpus=self.df["job_title"].to_list()
        def sp_tknizer(text): 
            return simple_preprocess(text, deacc=True)
        vectorizer=TfidfVectorizer(
            tokenizer=sp_tknizer, 
            preprocessor=None, 
            lowercase=False, 
            token_pattern=None 
        )
        self.tf_idf_fitted_=vectorizer.fit(self.corpus)
        vocab=vectorizer.vocabulary_
        self.tfidf_inv_vocab_={ind: token for token, ind in vocab.items()}
        print("tfidf fitted. ")
        
        self.data_=[sp_tknizer(x) for x in self.corpus]
        
        if fasttext_kwards is None: 
            self.fasttext_model_=FastText(
                sentences=self.data_, 
                vector_size=vector_size, 
                window=window,
                min_count=min_count, 
                sg=sg, 
                epochs=epochs, 
                min_n=min_n, 
                max_n=max_n, 
                bucket=bucket,
                workers=workers
            )
        else: 
            self.fasttext_model_=FastText(
                sentences=self.data_, 
                vector_size=vector_size, 
                window=window,
                min_count=min_count, 
                sg=sg, 
                epochs=epochs, 
                min_n=min_n, 
                max_n=max_n, 
                bucket=bucket,
                workers=workers, 
                **fasttext_kwards
            )
        self.vector_size=vector_size
        print("fasttest fitted. ")
        
        return self
        
    def _get_vec_token(self, 
                 token: str): 
        if token in self.fasttext_model_.wv: 
            return self.fasttext_model_.wv[token]
        else: 
            return None #np.zeros(shape=self.vector_size, dtype=np.float32)
        
    def _get_weighted_mean_sent(self, 
                                sent: list, 
                                weights):
        num=np.zeros(shape=self.vector_size, dtype=np.float32)
        den=0
        for word, wit in zip(sent, weights): 
            vec=self._get_vec_token(token=word)
            if (not vec is None) and (wit > 0): 
                num+= wit*vec 
                den+= wit 
                
        return num/den if den>0 else num 
                
    
    def _get_tfidf_weighted_emb_sents(self, 
                          sents: List[str]): 
        X=self.tf_idf_fitted_.transform(sents) 
        embs=np.zeros(shape=(X.shape[0], self.vector_size), dtype=np.float32)
        for i in range(X.shape[0]): 
            row = X.getrow(i) 
            idxs=row.indices
            vals=row.data 
            tokens=[self.tfidf_inv_vocab_[j] for j in idxs] 
            weights=vals 
            embs[i]=self._get_weighted_mean_sent(sent=tokens, weights=weights)
            
        return np.array(embs)
        
    def create_score(self, 
                     query: list, 
                     query_coef: Union[str, list]="evenly", 
                     coef_old: float=0.3, 
                     coef_new: float=0.7): 
        
        ft_fitted=hasattr(self, "fasttext_model_") 
        tfidf_fitted=hasattr(self, "tf_idf_fitted_")
        if (not ft_fitted) or (not tfidf_fitted): 
            ValueError(f"tfidf is fitted: {tfidf_fitted}. \n fasttext is fitted: {ft_fitted}.") 
        self.q_embs=self._get_tfidf_weighted_emb_sents(sents=query)
        print("Query embedding created. ")
        if not hasattr(self, "corpus_embs_"): 
            self.corpus_embs_=self._get_tfidf_weighted_emb_sents(sents=self.corpus)
            print("Corpus embedding created. ")
        else: 
            print("Using corpus embedding from the past. ")
        self.cos_sim=cosine_similarity(X=self.corpus_embs_, Y=self.q_embs)
        if query_coef=="evenly": 
            num_query=len(query)
            even_val=1/num_query
            query_coef=np.full(shape=(num_query, 1), fill_value=even_val)
        else: 
            query_coef=np.array(query_coef)[...,None]
        self.new_scores=(self.cos_sim @ query_coef).ravel()
        if not hasattr(self, "df_fitted_"):
            self.df_fitted_=self.df[["id", "job_title"]]
        #Order the df_fitted_ by id, ascendingly 
        self.df_fitted_=self.df_fitted_.sort_values(by="id") 
        #If there is already an score present, make it the old score. 
        if "fit_score" in self.df_fitted_.columns: 
            self.df_fitted_["old_score"]=self.df_fitted_["fit_score"] 
        #If no fit score is present, set old_score to 0. 
        else: 
            self.df_fitted_["old_score"]=0 
        #Set the new_scores 
        self.df_fitted_["new_score"]=self.new_scores 
        #Create the fit_score as a linear combination of the the old and new scores. If fit score does not exist, this would have been the first time this is done. 
        if "fit_score" in self.df_fitted_.columns: 
            self.df_fitted_["fit_score"]= coef_old * self.df_fitted_["old_score"] + coef_new * self.df_fitted_["new_score"] 
        else: 
            self.df_fitted_["fit_score"]=self.df_fitted_["new_score"] 
        return self 
            
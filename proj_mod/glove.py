import gensim.downloader as api
import os
from typing import Union
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

# os.environ["GENSIM_DATA_DIR"] = "../data/fitted/GloVe" 

class GloVeRanker: 
    def __init__(self):
        self.GloVe_fitted_=api.load("glove-wiki-gigaword-50")
        self.emb_dim_=self.GloVe_fitted_.vector_size 
        pass
    
    def fit(self, 
            df: pd.DataFrame):
        if hasattr(self, "df_fitted_"): 
            proceed=input("Fitted data is present, proceeding WILL OVERRIDE the fitted data. Please confirm. (y/n)").lower()
            if proceed=="n": 
                print("Fitting aborted. ") 
                return self 
            delattr(self, "df_fitted_")
            print("Fitted data removed. Proceeding...") 
        #Order the dfby id, ascendingly 
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
        self.tfidf_fitted_=vectorizer.fit(self.corpus)
        vocab=vectorizer.vocabulary_
        self.inv_vocab_={ind: token for token, ind in vocab.items()}
        
        return self
    
    def _return_vec(self, 
                   token: str): 
        if self.GloVe_fitted_.has_index_for(token): 
            return self.GloVe_fitted_.get_vector(token)
        else: 
            return None
    
    def _sent_weighted_mean(self, 
                            tokens: str, 
                            weights): 
        num=np.zeros(self.emb_dim_, dtype=np.float32)
        den=0
        for tok, wit in zip(tokens, weights): 
            vec=self._return_vec(tok)
            if (vec is not None) & (wit>0): 
                num += wit*vec 
                den += wit 
            
        return num/den if den>0 else num 
        
    
    def tfidf_sent_emb(self, 
                        sents: list): 
        X=self.tfidf_fitted_.transform(sents)
        embs=np.zeros(shape=(X.shape[0], self.emb_dim_), dtype=np.float32)
        for i in range (X.shape[0]): 
            row=X.getrow(i)
            idxs=row.indices
            vals=row.data
            tokens=[self.inv_vocab_[j] for j in idxs]
            weights=vals 
            embs[i]=self._sent_weighted_mean(tokens, weights)
        
        return np.array(embs) 
            
    
    def create_score(
                    self, 
                    query: list, 
                    query_coef: Union[str, list]="evenly", 
                    coef_old: float=0.3, 
                    coef_new: float=0.7
                    ): 
        if not hasattr(self, "GloVe_fitted_"): 
            ValueError("There is no fitted model, the initialization might have failed. ") 
        if not hasattr(self, "tfidf_fitted_"): 
            ValueError("TF-IDF sentence vectorizer is not fitted, please fit it first. ")
        #Process the query 
        self.q_vec_data_=self.tfidf_sent_emb(query) 
        #Process the corpus 
        self.vec_data_=self.tfidf_sent_emb(self.corpus)
        
        #Calculate cos similarity 
        self.cos_sim_=cosine_similarity(X=self.vec_data_, Y=self.q_vec_data_) #Shape (X n sample, Y n sample) = (104,2) in out context 
        #If query_ceof is "evenly" 
        if query_coef=="evenly": 
            num_query=len(query)
            even_val=1/num_query
            query_coef=np.full(shape=(num_query, 1), fill_value=even_val)
        else: 
            query_coef=np.array(query_coef)[..., None]
        self.new_scores_=(self.cos_sim_ @ query_coef).ravel()
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
        self.df_fitted_["new_score"]=self.new_scores_ 
        #Create the fit_score as a linear combination of the the old and new scores. If fit score does not exist, this would have been the first time this is done. 
        if "fit_score" in self.df_fitted_.columns: 
            self.df_fitted_["fit_score"]= coef_old * self.df_fitted_["old_score"] + coef_new * self.df_fitted_["new_score"] 
        else: 
            self.df_fitted_["fit_score"]=self.df_fitted_["new_score"] 
        return self 
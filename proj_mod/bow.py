import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from typing import Union

class BowRanker: 
    def __init__(
                self, 
                tfidf_kwargs: dict,
                sublinear_tf: bool=True, 
                use_idf: bool=True, 
                smooth_idf=True, 
                stop_word: str="english"
                ):
        self.vectorizer= TfidfVectorizer(norm="l2", sublinear_tf=sublinear_tf, use_idf=use_idf, smooth_idf=smooth_idf, stop_words=stop_word, **tfidf_kwargs)
        pass
    
    def fit(
            self, 
            df: pd.DataFrame
            ): 
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
        self.fitted_corpus_= self.vectorizer.fit_transform(self.corpus) 
        return self 
    
    def create_score(
                    self, 
                    query: list, 
                    query_coef: Union[str, list]="evenly", 
                    coef_old: float=0.3, 
                    coef_new: float=0.7
                    ): 
        if not hasattr(self, "fitted_corpus_"): 
            ValueError("There is no fitted corpus, please fit first. ") 
        self.fitted_query_=self.vectorizer.transform(query) 
        self.new_scores_=(self.fitted_corpus_ @ self.fitted_query_.T).toarray() #.ravel()
        #If query_ceof is "evenly" 
        if query_coef=="evenly": 
            num_query=len(query)
            even_val=1/num_query
            query_coef=np.full(shape=(num_query, 1), fill_value=even_val)
        else: 
            query_coef=np.array(query_coef)[..., None]
        self.new_scores_=(self.new_scores_ @ query_coef).ravel()
        #If df_fitted_ does not exist, create it 
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
        
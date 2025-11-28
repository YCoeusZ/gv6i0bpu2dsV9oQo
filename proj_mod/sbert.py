from sentence_transformers import SentenceTransformer, util 
from typing import Union 
import pandas as pd 
import numpy as np

class SBertRanker: 
    def __init__(self, 
                 model_name="all-MiniLM-L6-v2", device: str="cpu"):
        self.SBert_model_ = SentenceTransformer(model_name, device=device)
        pass
    
    def fit(self, 
            df: pd.DataFrame):
        self.df=df.sort_values(by="id")
        self.corpus=self.df["job_title"].to_list()
        return self
    
    def create_score(self, 
                    query: list, 
                    query_coef: Union[str, list]="evenly", 
                    coef_old: float=0.3, 
                    coef_new: float=0.7): 
        
        has_model=hasattr(self, "SBert_model_")
        if not has_model: 
            ValueError("Something is wrong, there is no trained model. ")
        self.q_embs=self.SBert_model_.encode(query, normalize_embeddings=True)
        print("Query embedding created. ")
        if not hasattr(self, "corpus_embs_"): 
            self.corpus_embs_=self.SBert_model_.encode(self.corpus, normalize_embeddings=True)
            print("Corpus embedding created. ")
        else: 
            print("Using corpus embedding from the past. ")
        self.cos_sim=np.array(util.cos_sim(self.corpus_embs_, self.q_embs).cpu())
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
            
import pandas as pd 
import numpy as np 
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from typing import Union
from sklearn.metrics.pairwise import cosine_similarity

class Word2VecRanker: 
    def __init__(
                self, 
                # vectorsize: int= 200, 
                # window: int = 5, 
                # sg: int=1, 
                # negative: int=10, 
                # min_count: int=5, 
                # epochs: int=5, 
                # workers: int=8
                ):
        pass

    def fit(self, 
            df: pd.DataFrame,
            w2v_kwargs: Union[dict,None]=None,
            vector_size: int= 200, 
            window: int = 5, 
            sg: int=1, 
            negative: int=10, 
            min_count: int=5, 
            epochs: int=5, 
            workers: int=8
            ):
        self.vector_size=vector_size
        if hasattr(self, "df_fitted_"): 
            proceed=input("Fitted data is present, proceeding WILL OVERRIDE the fitted data. Please confirm. (y/n)").lower()
            if proceed=="n": 
                print("Fitting aborted. ") 
                return self 
            delattr(self, "df_fitted_")
            print("Fitted data removed. Proceeding...") 
        #Order the dfby id, ascendingly 
        self.df=df.sort_values(by="id")
        corpus=self.df["job_title"].to_list()
        self.data_=[simple_preprocess(x) for x in corpus]
        if w2v_kwargs is None: 
            self.w2v_model_=Word2Vec(
                sentences=self.data_, 
                vector_size=vector_size, 
                window=window,
                sg=sg,
                negative=negative, 
                min_count=min_count,
                epochs=epochs,
                workers=workers
            )
        else: 
            self.w2v_model_=Word2Vec(
                sentences=self.data_, 
                vector_size=vector_size, 
                window=window,
                sg=sg,
                negative=negative, 
                min_count=min_count,
                epochs=epochs,
                workers=workers, 
                **w2v_kwargs
            )
        print("w2v model fitted. ")
        return self 
    
    def save_model(self, 
                   save_path: str
                   ): 
        if not hasattr(self, "w2v_model_"): 
            ValueError("No fitted w2v model, please fit first. ") 
        self.w2v_model_.save(save_path) 
        print("w2v model saved. ") 
        return self
    
    def load_model(self, 
                   load_path: str
                   ): 
        if hasattr(self, "w2v_model_"): 
            confirm=input("Fitted w2v model present, loading will override, please confirm. (y/n)").lower()
            if confirm == "n": 
                print("Loading aborted. ")
                return self
        self.w2v_model_=Word2Vec.load(load_path) 
        print("w2v model loaded. ") 
        
    def create_score(
                    self, 
                    query: list, 
                    query_coef: Union[str, list]="evenly", 
                    coef_old: float=0.3, 
                    coef_new: float=0.7
                    ): 
        if not hasattr(self, "w2v_model_"): 
            ValueError("There is no fitted model, please fit first. ") 
        #Process the query 
        #Apply simple_preprocess to each element of query 
        self.q_data_=[simple_preprocess(x) for x in query]
        #Remove anything in query that might not exist in wv 
        self.q_data_=[[token for token in title if (token in self.w2v_model_.wv)] for title in self.q_data_]
        #calculate the vector of each query as a vector 
        self.q_vec_data_=np.array([self.w2v_model_.wv[*title].sum(axis=0) if len(title) else np.zeros(shape=self.vector_size) for title in self.q_data_])
        
        #Process the corpus
        self.data_=[[token for token in title if (token in self.w2v_model_.wv)] for title in self.data_]
        self.vec_data_=np.array([self.w2v_model_.wv[*title].sum(axis=0) if len(title) else np.zeros(shape=self.vector_size) for title in self.data_ ])
        
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
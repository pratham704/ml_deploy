
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib 


app = FastAPI()

class ScoringItem(BaseModel):
    Age: int                        
    Gender: int                      
    self_employed: int               
    family_history: int                                
    work_interfere : int             
    no_employees: int       
    remote_work: int 
    tech_company: int 
    benefits: int 
    care_options: int 
    wellness_program: int 
    seek_help: int                   
    anonymity: int             
    leave: int       
    mental_health_consequence: int 
    phys_health_consequence: int 
    coworkers: int 
    supervisor: int 
    mental_health_interview: int 
    phys_health_interview: int 
    mental_vs_physical: int 
    obs_consequence: int 

with open('main_model_git.pkl','rb') as f:
    model=joblib.load(f)


@app.post('/')
async def scorinng_endpoints(item:ScoringItem):
    df=pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat=model.predict(df)
    return {"prediction": int(yhat)}
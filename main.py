import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel ,Field
from typing import Annotated ,Literal
from fastapi.middleware.cors import CORSMiddleware    # api ko frontend se jodne ke liye 


model =joblib.load('Mental_Health_Model.pkl')

app= FastAPI()

# API ko server se jodne ke liye 

app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# First pydantic model
class StudentData(BaseModel):
    age                         :int =Field(..., gt= 10 ,lt=100)
    gender                      :Literal['Male','Female',]
    country                     :str
    academic_level              :Literal['Undergraduate','Graduate', 'High School']
    most_used_platform          :Literal['Facebook','LinkedIn','Instagram','Snapchat','Twitter','YouTube','TikTok','LINE','KakaoTalk','VKontakte', 'WhatsApp', 'WeChat']
    purpose_of_use              :Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours       :float =  Field(..., gt=0 ,lt=24)
    daily_unlocks               :int   =  Field(..., gt=0)
    study_hours                 :float =  Field(...,gt=0 , lt=24)
    physical_activity_hours     :float =  Field(... , gt=0 , lt= 24)
    sleep_hours_per_night       :float =  Field(... , gt=0 ,lt= 24)
    stress_level                :Literal['Low','Medium',  'High', 'Very High']


# Describe what we send back
class PredictionResponse(BaseModel):
    predicted_Mental_health_score:float

@app.get('/')
def greet( ):
    return {'Welcome to Vivek Workshops '}

top_countries = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK','Germany', 'Mexico',
'Turkey','France']

@app.post('/predict' , response_model= PredictionResponse)

def predict (data : StudentData):
    
    country_group =data.country if data.country in top_countries else 'Other'


    input_row =pd.DataFrame([{
            'Age'                           :data.age, 
            'Gender'                        :data.gender,
            'Country'                       :data.country,
            'Academic_Level'                :data.academic_level,
            'Most_Used_Platform'            :data.most_used_platform,
            'Purpose_Of_Use'                :data.purpose_of_use,               
            'Avg_Daily_Usage_Hours'         :data.avg_daily_usage_hours,
            'Daily_Unlocks'                 :data.daily_unlocks,
            'Study_Hours'                   :data.study_hours,
            'Physical_Activity_Hours'       :data.physical_activity_hours,
            'Sleep_Hours_Per_Night'         :data.sleep_hours_per_night,
            'Stress_Level'                  :data.stress_level,
            'Grouped_Countries'             :country_group
            }])
        
    prediction = model.predict(input_row)[0]
    return PredictionResponse  (predicted_Mental_health_score = round(float(prediction),2))
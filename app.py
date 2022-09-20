# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:21:19 2022

@author: DELL
"""
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()


origins=["*"]

app.addle_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credencials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

class model_input(BaseModel):
    
    person_age                    :  int
    person_income                 :  int
    person_emp_length             :float
    loan_amnt                     :  int
    loan_int_rate                 :float
    loan_status                   :  int
    loan_percent_income           :float
    cb_person_cred_hist_length    :  int
    OTHER                         :  int
    OWN                           :  int
    RENT                          :  int
    EDUCATION                     :  int
    HOMEIMPROVEMENT               :  int
    MEDICAL                       :  int
    PERSONAL                      :  int
    VENTURE                       :  int
    B                             :  int
    C                             :  int
    D                             :  int
    E                             :  int
    F                             :  int
    G                             :  int
    
#Chargement du modèle enregisté

clients_model=pickle.load(open('model.pkl','rb'))

@app.post('/proba_defaut_client')  
def proba_defaut_pred(input_parameters:model_input):
    input_data=input_parameters.json()
    input_dictionary=json.loads(input_data)
    
    
    persag=input_dictionary[' person_age' ]
    persin=input_dictionary['person_income']
    persem=input_dictionary[' person_emp_length' ]
    loaam=input_dictionary['loan_amnt' ]
    loaint=input_dictionary['  loan_int_rate' ]
    loast=input_dictionary['loan_status' ]  
    loape=input_dictionary[' loan_percent_income' ]
    cbper=input_dictionary[' cb_person_cred_hist_length' ]
    oth=input_dictionary['  OTHER' ]
    own=input_dictionary['  OWN' ]
    ren=input_dictionary[' RENT' ]
    edu=input_dictionary['  EDUCATION' ]
    homei=input_dictionary[' HOMEIMPROVEMENT' ]
    med=input_dictionary['  MEDICAL'] 
    pers=input_dictionary['PERSONAL' ]
    vent=input_dictionary[' VENTURE' ]
    b= input_dictionary[' B' ]
    c=input_dictionary[' C' ]
    d= input_dictionary[' D' ]
    e= input_dictionary[' E' ]
    f=input_dictionary['F' ]    
    g= input_dictionary['G' ]

    input_list =[persag, persin, persem,loaam, loaint , loast
  , loape, cbper, oth,own, ren, edu, homei, med, pers, vent,b,c,d,e,f,g]

    prediction=clients_model.predict([input_list])
    
    if prediction[0]==0:
        return 'Le client n''est pas solvable'
    else:
        return 'Le cclient est solvable'
    
    
    
#!/usr/bin/env python
# coding: utf-8

import requests

url = "http://localhost:9696/predict"

customer = {
    "customerid": "8879-zkjof",
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "tenure": 41,
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "no",
    "deviceprotection": "yes",
    "techsupport": "yes",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "one_year",
    "paperlessbilling": "yes",
    "paymentmethod": "bank_transfer_(automatic)",
    "monthlycharges": 79.85,
    "totalcharges": 3320.75
}


response = requests.post(url, json=customer).json()

if response['Churn'] == True:
    print("We're sending promo to customer id: {}".format('xyz-123'))
elif response['Churn'] == False:
    print("We're not sending a promo")





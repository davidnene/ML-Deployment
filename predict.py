import pickle

print("load model")
model_file = 'churn-model.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

print(f' artefacts: {dv, model}')


customer = {
    'customerid': '8879-zkjof',
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 41,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'yes',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 79.85,
    'totalcharges': 3320.75
}

X = dv.transform([customer])
print(f'Model prediction: Churn probability: {model.predict_proba(X)[0, 1]}')


import pickle
from flask import Flask
from flask import request
from flask import jsonify

print("load model")

model_file = 'churn-model.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

print(f' artefacts: {dv, model}')



app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    results = {
        "Churn Probability": float(y_pred.round(2)),
        "Churn": bool(churn)
    }
    
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
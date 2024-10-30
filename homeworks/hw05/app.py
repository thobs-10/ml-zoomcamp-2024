import joblib
from flask import Flask
from flask import request
from flask import jsonify
from waitress import serve

model_path = 'model1.bin'
dict_vectorizer_path = 'dv.bin'

model = joblib.load(open(model_path, 'rb'))
dict_vectorizer = joblib.load(open(dict_vectorizer_path, 'rb'))

def predict_client(data: dict) -> float:
    X = dict_vectorizer.transform(data)
    y = model.predict_proba(X)[0, 1]
    return float(y)  

app = Flask("churn-prediction")

@app.route("/predict", methods=["POST"]) 
def predict():
    customer = request.get_json()
    
    if not customer:
        return jsonify({"error": "No data provided"}), 400

    prediction = predict_client(customer)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=9696)
    # app.run(debug=True, host='0.0.0.0', port=9696)
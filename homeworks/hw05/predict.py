import joblib

model_path = 'C:/Users/Thobs/Desktop/Portfolio/Projects/Data-Science-Projects/ml-zoomcamp-2024/homeworks/artifacts/model1.bin'
dict_vectorizer = 'C:/Users/Thobs/Desktop/Portfolio/Projects/Data-Science-Projects/ml-zoomcamp-2024/homeworks/artifacts/dv.bin'

model = joblib.load(open(model_path, 'rb'))
dict_vectorizer = joblib.load(open(dict_vectorizer, 'rb'))

client ={
    "job": "management",
    "duration": 400,
    "poutcome": "success"
}

def predict(data: dict) -> float:
    X = dict_vectorizer.transform(data)
    y = model.predict_proba(X)[0, 1]
    return y

if __name__ == "__main__":
    print(predict(client))
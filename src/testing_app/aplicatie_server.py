from flask import Flask, request, jsonify
import numpy as np
import joblib




app = Flask(__name__)

# Incarcarea modelelelor antrenate

model_abclf = joblib.load('model_abclf.pkl')
model_bclf = joblib.load('model_bclf.pkl')
model_dtclf = joblib.load('model_dtclf.pkl')
model_rfclf = joblib.load('model_rfclf.pkl')
model_etclf = joblib.load('model_etclf.pkl')
model_gbclf = joblib.load('model_gbclf.pkl')
model_hgbclf = joblib.load('model_hgbclf.pkl')



# Definirea endpoint-ului pentru serverul de predictie

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    model = data['model']

    if model == 'AdaBoostClassifier':
        prediction = model_abclf.predict(features)
    elif model == 'BaggingClassifier':
        prediction = model_bclf.predict(features)
    elif model == 'DecisionTreeClassifier':
        prediction = model_dtclf.predict(features)
    elif model == 'RandomForestClassifier':
        prediction = model_rfclf.predict(features)
    elif model == 'ExtraTreesClassifier':
        prediction = model_etclf.predict(features)
    elif model == 'GradientBoostingClassifier': 
        prediction = model_gbclf.predict(features)
    elif model == 'HistGradientBoostingClassifier':
        prediction = model_hgbclf.predict(features)
    else:
        prediction = '0'

    return jsonify({
        'prediction': int(prediction[0]) if isinstance(prediction, np.ndarray) else prediction
    })

if __name__ == '__main__':
    app.run(debug=True)

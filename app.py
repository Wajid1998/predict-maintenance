from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_placement():
    Type = int(request.form.get('Type'))
    Air_Temperature = float(request.form.get('Air_Temperature'))
    Process_Temperature = float(request.form.get('Process_Temperature'))
    Rotational_Speed = int(request.form.get('Rotational_Speed'))
    Torque = float(request.form.get('Torque'))
    Tool_wear = int(request.form.get('Tool_wear'))
    TWF = int(request.form.get('TWF'))
    HDF = int(request.form.get('HDF'))
    PWF = int(request.form.get('PWF'))
    OSF = int(request.form.get('OSF'))
    RNF = int(request.form.get('RNF'))




    # prediction
    result = model.predict(np.array([Type, Air_Temperature, Process_Temperature, Rotational_Speed, Torque, Tool_wear, TWF, HDF, PWF, OSF, RNF]).reshape(1, 11))

    if result[0] == 1:
        result = 'Failure'
    else:
        result = 'Not Failure'

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

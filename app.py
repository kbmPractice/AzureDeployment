import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('iris_model.pkl','rb'))



@app.route('/')
def home():
    return render_template('iris_index.html')



@app.route('/predict',methods =['POST'])
def predict():
    int_features = [float(x)for x in request.form.values()]
    final_features =[np.array(int_features)]
    prediction =model.predict(final_features)
    #print(prediction[0])
    iris_names= ['Setosa','Versicolor','Virginica']
    output = iris_names[int(prediction[0])]
    return render_template('iris_index.html',prediction_text ="IRIS class is {}".format(output))
    
@app.route('/predict_api',methods =['POST'])  
def predict_api():
    '''
        For direct API calls through request
    
    '''
    
    data = request.get_json(force = True)
    prediction = model. predict([np.array(list(data.values()))])
    
    output = prediction[0]
    return jsonify (output)


if __name__ == '__main__':
    app.run(debug = True)
    
    
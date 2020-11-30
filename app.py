import json
import joblib
from flask import Flask, jsonify, abort, make_response, request
#from sklearn.externals import joblib
import numpy as np

# Create Flask object to run
app = Flask(__name__, template_folder="templates")  

iris_model = joblib.load('model/iris_model.pkl')

@app.route('/')
@app.route("/home")
@app.route("/index")
def index():
    return make_response(
        jsonify({'message': 'Iris model deployment using flask.'}),
        200
    )  

@app.route("/predict", methods=['POST'])
def predict():
    if not request.json:
        abort(400)
    if not 'sepal_length' in request.json :
        abort(400)
    if not 'sepal_width' in request.json :
        abort(400)
    if not 'petal_length' in request.json :
        abort(400)
    if not 'petal_width' in request.json :
        abort(400)

    # read data from body of the request
    test_data = request.get_json(force=True)


    #sepal_length = test_data['sepal_length']
    #sepal_width = test_data['sepal_width']
    #petal_length = test_data['petal_length']
    #petal_width = test_data['petal_width']


    #test_inp_1 = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, 4)
    
    # convert to numpy array
    test_inp = np.array(tuple(test_data.values())).reshape(1, 4)


    class_predicted = int(iris_model.predict(test_inp)[0])
    output = str(class_predicted)

    return make_response(
        jsonify({'Predicted Iris Class': output}),
        200
    ) 






@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad Request'}), 400)


if __name__ == "__main__":
    # Start Application
    app.run()
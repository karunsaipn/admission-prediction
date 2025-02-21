from flask import Flask,render_template,request,jsonify
from flask_cors import CORS, cross_origin
import pickle

app = Flask(__name__, static_folder='static', template_folder='templates')

CORS(app)

try:
    filename = 'chance_of_admission_model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None


@app.route('/', methods=['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')



@app.route('/predict', methods=['POST', 'GET'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            # Reading inputs given by the user
            GRE_Score = float(request.form['GRE_Score'])
            TOEFL_Score= float(request.form['TOEFL_Score'])
            University_Rating= float(request.form['University_Rating'])
            SOP= float(request.form['SOP'])
            LOR= float(request.form['LOR'])
            CGPA= float(request.form['CGPA'])

            filename = 'chance_of_admission_model.pkl'
            loaded_model = pickle.load(open(filename, 'rb'))

            prediction = loaded_model.predict([[GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA]])
            print('Prediction is: ', prediction)

            return render_template('result.html', prediction=round(100 * prediction[0]))

        except Exception as e:
            print('The Exception is: ', e)
            return 'Something is wrong bro, use your expensive brain bro'
    else:
        return render_template('index.html')

@app.route('/test-logo')
def test_logo():
    return app.send_static_file('logo.png')    @app.route('/test-logo')
    def test_logo():
        return app.send_static_file('logo.png')
if __name__ == "__main__":
    app.run(debug=True)
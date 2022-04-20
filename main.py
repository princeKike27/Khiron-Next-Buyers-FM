import numpy as np
import sklearn
import pickle
from flask import Flask, request, jsonify, render_template, url_for


''' FLASK
Light Web Framework that allows you to build web applications
'''
# initialize flask
app = Flask(__name__)

# read model file in binary >> rb
with open('k_next_buyers.pickle', 'rb') as f:
    model = pickle.load(f)


'''***********************************************************************'''
''' ENDPOINTS & ROUTINES
HTTP >> request-response protocol to connect between a client & server
GET >> request data from a specified source (ex: web page form)
POST >> data sent to the server to create/update a resource (ex: web page element)
'''

'''HOME PAGE'''
@app.route('/')
def home():
    # render html page
    return render_template('home.html')



'''PREDICT BUYER'''
@app.route('/predict', methods=['POST'])
def predict():
    # save values from form
    gender_f = request.form.get('gender_form')
    age_f = request.form.get('age_form')
    plan_f = request.form.get('plan_form')
    product_f = request.form.get('product_form')
    city_f = request.form.get('city_form')
    diagnostic_f = request.form.get('diagnostic_form')
    exam_f = request.form.get('exam_form')

    # print patient features
    print('\n')
    print(f'Patient - gender:{gender_f}, age:{age_f}, plan:{plan_f}, product:{product_f}, city:{city_f}, diagnostic_f:{diagnostic_f}, exam:{exam_f}')

    # encode values from form to be fed into the model
    sex = 1 if gender_f == 'Masculino' else 0
    is_51_76 = 1 if age_f == '51-75' else 0
    is_76_101 = 1 if age_f == '76-101' else 0
    is_particular = 1 if plan_f == 'particular' else 0
    is_fm002 = 1 if product_f == 'fm002' else 0
    is_fm001 = 1 if product_f == 'fm001' else 0
    is_medellin = 1 if city_f == 'medellin' else 0
    is_mosquera = 1 if city_f == 'mosquera' else 0
    is_bogota = 1 if city_f == 'bogota' else 0
    is_cronic_pain = 1 if diagnostic_f == 'dolor_cronico' else 0
    is_anxiety_depres = 1 if diagnostic_f == 'anxiedad_depre' else 0
    is_cronic_pain_uncu = 1 if diagnostic_f == 'dolor_cronico_incu' else 0
    is_control = 1 if exam_f == 'Control' else 0

    # print patient encoded features
    print(f'Patient Encoded - sex:{sex}, 51-76:{is_51_76}, 76-101:{is_76_101}, particular:{is_particular}, fm002:{is_fm002}, fm001:{is_fm001}, medellin:{is_medellin}, mosquera:{is_mosquera}, bogota:{is_bogota}, cronic_pain:{is_cronic_pain}, anxiety:{is_anxiety_depres}, cronic_pain_uncu:{is_cronic_pain_uncu}, control:{is_control}', '\n')

    # save encoded features in np array
    features = np.array([sex, is_51_76, is_76_101, is_particular, is_fm002, is_fm001, is_medellin, is_mosquera, is_bogota, is_cronic_pain, is_anxiety_depres, is_cronic_pain_uncu, is_control])
    features = features.reshape(1, -1)
    print(f'Encoded Features: {features}', '\n')


    '''PREDICTION'''
    # prediction with classification threshold >= 0.675 (1 == Buyer, 0 == Non_Buyer)
    buyer = 0
    prediction = model.predict_proba(features)
    print(f'Probabilities: {prediction}')
    print(f'Patient Prediction: {prediction[0][1]:.2f}')


    # render html page according to prediction
    if prediction[0][1] >= 0.675:
        buyer = 1
        print(f'With a Probability of {prediction[0][1]:.2f} the patient is a Buyer!!!', '\n')
        return render_template('home.html', prediction_text='El paciente es un comprador potencial de cannabis')
    else:
        print(f'With a Probability of {prediction[0][1]:.2f} the patient is Not a Buyer!!!', '\n')
        return render_template('home.html', prediction_text='El paciente no es un comprador potencial de cannabis')


'''RUN SERVER'''
if __name__ == '__main__':
    print('*' * 100)
    print('Starting Python Flask Server for K Next Nuyers ...', '\n')
    print(f'Intercept: {model.intercept_[0]}')
    print(f'Coefficients: {model.coef_[0]}', '\n')

    # run server
    app.run(debug=True)

import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

df=pd.read_csv("data.csv")

@app.route("/")
def loadPage():
	return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    
    '''
        1       gender                                       int64
        2       SeniorCitizen                                int64
        3       Partner                                      int64
        4       Dependents                                   int64
        5       tenure                                     float64
        6       PhoneService                                 int64
        7       MultipleLines                                int64
        8       OnlineSecurity                               int64
        9       OnlineBackup                                 int64
        10      DeviceProtection                             int64
        11      TechSupport                                  int64
        12      StreamingTV                                  int64
        13      StreamingMovies                              int64
        14      PaperlessBilling                             int64
        15      MonthlyCharges                             float64
        16      TotalCharges                               float64
        17_1    InternetService_DSL                          uint8
        17_2    InternetService_Fiber optic                  uint8
        17_3    InternetService_No                           uint8
        18_1    Contract_Month-to-month                      uint8
        18_2    Contract_One year                            uint8
        18_3    Contract_Two year                            uint8
        19_1    PaymentMethod_Bank transfer (automatic)      uint8
        19_2    PaymentMethod_Credit card (automatic)        uint8
        19_3    PaymentMethod_Electronic check               uint8
        19_4    PaymentMethod_Mailed check                   uint8
    '''
    
    # can shorten code
    # by making use of the fact that request.form 
    # is a(n) (immutable) dictionary!
    # so just loop over the dict!

    input1 = int(request.form['input1'] == 'Female')      # gives Male:0, Female:1
    input2 = int(request.form['input2'] == 'Yes')
    input3 = int(request.form['input3'] == 'Yes')
    input4 = int(request.form['input4'] == 'Yes')
    input5 = (float(request.form['input5']) - 1.0) / (72.0 - 71.0)    # tenure
    input6 = int(request.form['input6'] == 'Yes')
    input7 = int(request.form['input7'] == 'Yes')
    input8 = int(request.form['input8'] == 'Yes')
    input9 = int(request.form['input9'] == 'Yes')
    input10 = (request.form['input10'] == 'Yes')
    input11 = (request.form['input11'] == 'Yes')
    input12 = (request.form['input12'] == 'Yes')
    input13 = (request.form['input13'] == 'Yes')
    input14 = (request.form['input14'] == 'Yes')
    input15 = (float(request.form['input15']) - 18.25) / (118.75 - 18.25)
    input16 = (float(request.form['input16']) - 18.8) / (8684.8 - 18.8)
    # input17 = request.form['input17']
    input17_1 = (request.form['input17'] == 'InternetService_DSL')
    input17_2 = (request.form['input17'] == 'InternetService_Fiber optic')
    input17_3 = (request.form['input17'] == 'InternetService_No')
    # input18 = request.form['input18']
    input18_1 = (request.form['input18'] == 'Contract_Month-to-month')
    input18_2 = (request.form['input18'] == 'Contract_One year')
    input18_3 = (request.form['input18'] == 'Contract_Two year')
    # input19 = request.form['input19'] == 'PaymentMethod'
    input19_1 = (request.form['input19'] == 'PaymentMethod_Bank transfer (automatic)')
    input19_2 = (request.form['input19'] == 'PaymentMethod_Credit card (automatic)')
    input19_3 = (request.form['input19'] == 'PaymentMethod_Electronic check')
    input19_4 = (request.form['input19'] == 'PaymentMethod_Mailed check')

    model = keras.models.load_model('model_saved.h5')
    
    data = [
                [
                    input1,
                    input2,
                    input3,
                    input4,
                    input5,
                    input6,
                    input7,
                    input8,
                    input9,
                    input10,
                    input11,
                    input12,
                    input13,
                    input14,
                    input15,
                    input16,
                    # input17,
                    # input18,
                    # input19
                    input17_1,
                    input17_2,
                    input17_3,
                    input18_1,
                    input18_2,
                    input18_3,
                    input19_1,
                    input19_2,
                    input19_3,
                    input19_4
                ]
            ]
    
    pred = model.predict(data)[0,0]

    if pred>0.5:
        pred1 = 'Likely to churn'
        prob2 = str(round(pred*100,2))
    else:
        pred1 = 'Likely not to churn'
        prob2 = str(round((1-pred)*100,2))
        
    return render_template('result.html', pred1=pred1, prob2=prob2, 
                           input1 = request.form['input1'], 
                           input2 = request.form['input2'],
                           input3 = request.form['input3'],
                           input4 = request.form['input4'],
                           input5 = request.form['input5'], 
                           input6 = request.form['input6'], 
                           input7 = request.form['input7'], 
                           input8 = request.form['input8'], 
                           input9 = request.form['input9'], 
                           input10 = request.form['input10'], 
                           input11 = request.form['input11'], 
                           input12 = request.form['input12'], 
                           input13 = request.form['input13'], 
                           input14 = request.form['input14'], 
                           input15 = request.form['input15'], 
                           input16 = request.form['input16'], 
                           input17 = request.form['input17'],
                           input18 = request.form['input18'], 
                           input19 = request.form['input19'])

if __name__=="__main__":
    app.run(debug=True)
from django.shortcuts import render , HttpResponse
import pickle
import pandas as pd
import numpy as np

# Create your views here.

def index(request):
   # return HttpResponse("this is home page")
   return render(request,'index.html')

def result(request):
    a = int(request.GET['gender'])
    b = int(request.GET['age'])
    c = float(request.GET['height'])
    d = float(request.GET['weight'])
    e = float(request.GET['duration'])
    f = float(request.GET['heartrate'])
    g = float(request.GET['temp'])

    model = pickle.load(open('ml.sav', 'rb'))

    input_data = (a,b,c,d,e,f,g)

    # changing input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    input_data_reshaped=pd.DataFrame(input_data_reshaped)

    input_data_reshaped.columns=['Gender','Age','Height','Weight','Duration','Heart_Rate','Body_Temp']

    prediction = model.predict(input_data_reshaped)

    result = prediction[0]

    return render(request,'result.html',{'result': result})

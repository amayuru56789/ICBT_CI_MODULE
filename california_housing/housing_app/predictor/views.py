from django.shortcuts import render
import joblib
import numpy as np
from django.views import View
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Create your views here.

class PredictView(View):
    def get(self, request):
        return render(request, 'predictor/index.html')

def post(self, request):
        # Get form data
        features = [
            float(request.POST['MedInc']),
            float(request.POST['HouseAge']),
            float(request.POST['AveRooms']),
            float(request.POST['AveBedrms']),
            float(request.POST['Population']),
            float(request.POST['AveOccup']),
            float(request.POST['Latitude']),
            float(request.POST['Longitude']),
        ]
        
        # Load model and scaler (pretrained)
        model = joblib.load('predictor/model/xgb_model.joblib')
        scaler = joblib.load('predictor/model/scaler.joblib')
        
        # Preprocess and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        
        context = {
            'prediction': f"${prediction * 100000:,.2f}",
            'input_data': request.POST
        }
        return render(request, 'predictor/results.html', context)
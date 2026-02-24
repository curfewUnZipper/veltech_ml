curl -X POST http://127.0.0.1:5000/predict ^
-H "Content-Type: application/json" ^
-d "{\"voltage_rest\":12.1,
     \"voltage_load\":11.7,
     \"voltage_sag\":0.4,
     \"current\":8,
     \"temperature\":42,
     \"dv_dt\":-0.06}"


some tests:
{
  "voltage_rest": 12.6,
  "voltage_load": 12.3,
  "voltage_sag": 0.3,
  "current": 4.5,
  "temperature": 30,
  "dv_dt": -0.01
}


{
  "voltage_rest": 12.1,
  "voltage_load": 11.7,
  "voltage_sag": 0.4,
  "current": 7.5,
  "temperature": 38,
  "dv_dt": -0.04
}

{
  "voltage_rest": 11.4,
  "voltage_load": 10.8,
  "voltage_sag": 0.7,
  "current": 12,
  "temperature": 48,
  "dv_dt": -0.08
}


{
  "voltage_rest": 10.8,
  "voltage_load": 9.7,
  "voltage_sag": 1.2,
  "current": 16,
  "temperature": 55,
  "dv_dt": -0.12
}


{
  "voltage_rest": 12.5,
  "voltage_load": 12.2,
  "voltage_sag": 0.3,
  "current": 5,
  "temperature": 50,
  "dv_dt": -0.015
}

--------------------

Model Features:
voltage_rest
voltage_load
voltage_sag ⭐ (very predictive)
current
temperature
dv_dt ⭐
label



(venv) C:\Users\cryin\Work\Veltech\ml1>python train_rf_model.py
Dataset shape: (5000, 7)
   voltage_rest  voltage_load  voltage_sag   current  temperature     dv_dt  label
0     12.564482     12.307665     0.240992  7.183793    25.187507 -0.008203      0
1     11.864449     11.798600     0.122399  4.959937    34.013448 -0.011967      0
2     12.777187     12.750703     0.010715  0.858085    21.182515 -0.011645      0
3     12.674276     12.470086     0.244122  7.407552    32.857687 -0.009286      0
4     12.701037     12.609890     0.080250  3.140754    21.053572 -0.008700      0

✅ Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       900
           1       1.00      1.00      1.00       100

    accuracy                           1.00      1000
   macro avg       1.00      1.00      1.00      1000
weighted avg       1.00      1.00      1.00      1000


✅ Confusion Matrix:
[[900   0]
 [  0 100]]

🎯 Accuracy: 1.0000

🔥 Feature Importance:
voltage_load    0.333022
voltage_rest    0.304377
voltage_sag     0.186974
dv_dt           0.128528
current         0.042866
temperature     0.004233
dtype: float64
📊 Feature importance plot saved

💾 Model saved as battery_rf_model.pkl
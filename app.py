from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model dan scaler yang sudah dibuat tadi
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil inputan user dari form HTML
    data = {
        'Pregnancies': int(request.form['pregnancies']),
        'Glucose': int(request.form['glucose']),
        'BloodPressure': int(request.form['blood_pressure']),
        'SkinThickness': int(request.form['skin_thickness']),
        'Insulin': int(request.form['insulin']),
        'BMI': float(request.form['bmi']),
        'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree']),
        'Age': int(request.form['age'])
    }
    
    # Ubah format menjadi DataFrame seperti saat training
    input_data = pd.DataFrame(data, index=[0])
    
    # Transformasi data menggunakan scaler
    input_data_scaled = scaler.transform(input_data)
    
    # Prediksi hasilnya
    prediction = model.predict(input_data_scaled)[0]
    
    # Tentukan output teks
    if prediction == 1:
        hasil = 'Diabetic (Terindikasi Diabetes)'
    else:
        hasil = 'Non-Diabetic (Tidak Terindikasi Diabetes)'
        
    return render_template('index.html', prediction=hasil)

if __name__ == '__main__':
    app.run(debug=True)
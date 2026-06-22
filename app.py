from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import pandas as pd
import numpy as np
# Import library Dash
from dash import Dash, html, dcc, dash_table
import plotly.express as px

app = Flask(__name__)
app.secret_key = "arya_rahasia_123" # Kunci rahasia untuk sistem login (session)

# Load Model ML kamu
model = pickle.load(open('model.pkl', 'rb'))

# ==========================================
# 1. SETUP DASHBOARD (PLOTLY DASH)
# ==========================================
# Menjadikan Flask sebagai server induk untuk Dash
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

# Data dummy untuk grafik seperti foto dari dosenmu
df_dash = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})
fig = px.bar(df_dash, x="Fruit", y="Amount", color="City", barmode="group")

# Layout halaman Dashboard
dash_app.layout = html.Div(children=[
    html.H2(children='Dash Plotly - Demo RS Arya'),
    html.A("🔙 Kembali ke Halaman Prediksi", href="/predict", style={'color': 'blue', 'textDecoration': 'none'}),
    html.Br(), html.Br(),

    # --- INI TAMBAHAN TABELNYA ---
    html.Div([
        dash_table.DataTable(
            data=df_dash.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df_dash.columns],
            style_cell={'textAlign': 'left', 'padding': '5px', 'fontFamily': 'Arial'},
            style_header={'backgroundColor': 'white', 'fontWeight': 'bold', 'borderBottom': '1px solid black'}
        )
    ], style={'width': '80%', 'margin': 'auto'}), # Mengatur lebar tabel agar rapi

    html.Br(), html.Br(),
    
    # --- INI GRAFIKNYA ---
    html.Div(children='Amount by Fruit & City', style={'textAlign': 'left', 'width': '80%', 'margin': 'auto', 'fontSize': '20px'}),
    dcc.Graph(id='example-graph', figure=fig)
    
], style={'textAlign': 'center', 'fontFamily': 'Arial'})

# ==========================================
# 2. SETUP RUTE FLASK (LOGIN & PREDIKSI)
# ==========================================

# Halaman Pertama: Login
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Hardcode cek username dan password
        if username == 'admin' and password == 'admin123':
            session['logged_in'] = True
            return redirect(url_for('predict_page'))
        else:
            return render_template('login.html', error="Username atau Password Salah!")
            
    return render_template('login.html')

# Halaman Kedua: Prediksi Diabetes (Halaman HTML lama kamu)
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    # Cek apakah sudah login, kalau belum lempar ke halaman login
    if 'logged_in' not in session:
        return redirect(url_for('login'))
        
    if request.method == 'POST':
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
            input_data = pd.DataFrame(data, index=[0])
            scaler = pickle.load(open('scaler.pkl', 'rb'))
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)[0]
            hasil = 'Diabetic (Terindikasi Diabetes)' if prediction == 1 else 'Non-Diabetic (Tidak Terindikasi Diabetes)'
            return render_template('index.html', prediction=hasil)
        
    return render_template('index.html')

# Fitur Logout
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
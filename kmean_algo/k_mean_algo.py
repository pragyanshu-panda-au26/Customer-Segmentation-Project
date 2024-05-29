import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime, timedelta
from threading import Thread

app = Flask(__name__)
model = pickle.load(open('./kmeans_model.pkl', 'rb'))

def load_and_clean_data(file_path):
    retail = pd.read_csv(file_path, sep=',', encoding="ISO-8859-1", header=0)
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['CustomerID'] = retail['CustomerID'].apply(lambda x: x.split('.')[0])
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']
    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%m/%d/%y %H:%M')
    max_date = max(retail['InvoiceDate'])
    retail['Diff'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    rfm_p.columns = ['CustomerID', 'Recency']
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']
    Q1 = rfm.Amount.quantile(0.05)
    Q3 = rfm.Amount.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Amount >= Q1 - 1.5 * IQR) & (rfm.Amount <= Q3 + 1.5 * IQR)]
    Q1 = rfm.Recency.quantile(0.05)
    Q3 = rfm.Recency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Recency >= Q1 - 1.5 * IQR) & (rfm.Recency <= Q3 + 1.5 * IQR)]
    Q1 = rfm.Frequency.quantile(0.05)
    Q3 = rfm.Frequency.quantile(0.95)
    IQR = Q3 - Q1
    rfm = rfm[(rfm.Frequency >= Q1 - 1.5 * IQR) & (rfm.Frequency <= Q3 + 1.5 * IQR)]
    return rfm

def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    rfm['CustomerID'] = rfm['CustomerID'].astype(int)
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
    return rfm, rfm_df_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join(os.getcwd(), secure_filename(file.filename))
    file.save(file_path)
    rfm, df = preprocess_data(file_path)
    result_df = model.predict(df)
    rfm['Cluster_Id'] = result_df
    excel_file_path = os.path.join('static', 'rfm_clusters.xlsx')
    rfm[['CustomerID', 'Cluster_Id']].to_excel(excel_file_path, index=False)
    sns.stripplot(x='Cluster_Id', y='Amount', data=rfm, hue='Cluster_Id')
    amount_img_path = 'static/ClusterID_Amount.png'
    plt.savefig(amount_img_path)
    plt.clf()
    sns.stripplot(x='Cluster_Id', y='Frequency', data=rfm, hue='Cluster_Id')
    freq_img_path = 'static/ClusterID_Frequency.png'
    plt.savefig(freq_img_path)
    plt.clf()
    sns.stripplot(x='Cluster_Id', y='Recency', data=rfm, hue='Cluster_Id')
    recency_img_path = 'static/ClusterID_Recency.png'
    plt.savefig(recency_img_path)
    plt.clf()
    response = {
        'amount_img': amount_img_path,
        'freq_img': freq_img_path,
        'recency_img': recency_img_path,
        'excel_file': excel_file_path
    }
    return jsonify(response)

@app.route('/download_excel')
def download_excel():
    file_path = request.args.get('file')
    return send_file(file_path, as_attachment=True)

def delete_old_files():
    while True:
        current_time = datetime.now()
        folder = os.getcwd()  # This will cover all files in the current working directory
        
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            
            # Check if the file is a relevant temporary file
            if (os.path.isfile(file_path) and 
                (filename.endswith('.png') or filename.endswith('.xlsx') or filename.endswith('.csv'))):
                
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if current_time - file_time > timedelta(minutes=3):
                    os.remove(file_path)
        
        time.sleep(60)  # Sleep for 60 seconds before checking again


Thread(target=delete_old_files, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=True)

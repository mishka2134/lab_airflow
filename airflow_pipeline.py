import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta
from train_model import train  

def download_data():
    url = "https://raw.githubusercontent.com/mishka2134/lab1/main/energy_data.csv"
    df = pd.read_csv(url)
    df.to_csv("energy_raw.csv", index=False)
    print(f"Downloaded data shape: {df.shape}")
    return df

def clear_data():
    df = pd.read_csv("energy_raw.csv").dropna()
    cats = ['Building Type', 'Day of Week']
    nums = ['Square Footage', 'Number of Occupants', 
            'Appliances Used', 'Average Temperature', 'Energy Consumption']
    df = df[(df['Square Footage'] > 0) & (df['Square Footage'] <= 1e6)]
    df = df[(df['Number of Occupants'] > 0) & (df['Number of Occupants'] <= 1000)]
    df = df[(df['Average Temperature'] >= -30) & (df['Average Temperature'] <= 50)]
    df = df[(df['Energy Consumption'] > 0) & (df['Energy Consumption'] <= 1e6)]
    df = df[(df['Appliances Used'] >= 0) & (df['Appliances Used'] <= 100)]
  
    df[cats] = OrdinalEncoder().fit_transform(df[cats])
    df.to_csv('energy_cleaned.csv', index=False)
    print(f"Cleaned data shape: {df.shape}")
    return True

with DAG(
    dag_id="energy_pipeline",
    start_date=datetime(2025, 2, 3),
    schedule=timedelta(minutes=5),  
    max_active_tasks=4,
    max_active_runs=1,
    catchup=False,
) as dag_energy:

    download_task = PythonOperator(
        task_id="download_energy_data",
        python_callable=download_data,
    )

    clear_task = PythonOperator(
        task_id="clean_energy_data",
        python_callable=clear_data,
    )

    train_task = PythonOperator(
        task_id="train_energy_model",
        python_callable=train,
    )
    
    download_task >> clear_task >> train_task

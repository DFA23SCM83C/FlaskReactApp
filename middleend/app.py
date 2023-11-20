import base64
from datetime import datetime, timedelta, timezone
import json
import pytz
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import time
from collections import defaultdict
import requests
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import requests
import json
from google.cloud import storage
import pandas as pd

app = Flask(__name__)
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

CORS(app)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

appurl = os.environ['URL']

bucketurl = 'https://storage.googleapis.com/lstm-middleend-bucket/'


def upload_blob_from_memory(source_file_object, destination_blob_name):
    """Uploads a file to the bucket."""
    client = storage.Client()
    bucket = client.bucket('lstm-middleend-bucket')
    blob = bucket.blob(destination_blob_name)

    # Upload the image from the in-memory file
    blob.upload_from_file(source_file_object, content_type='image/png')

    print(f"File uploaded to {destination_blob_name}.")


@app.route('/api/plot/issues', methods=['GET'])
def get_data_issues_count():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/issues?repo=' + repo)
    data = response_API.text
    jsondata = json.loads(data)
    repo_name = list(jsondata.keys())[0]
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in jsondata[repo_name].keys()]
    counts = list(jsondata[repo_name].values())

    # Creating a line chart
    plt.plot(dates, counts, marker='o', linestyle='-')

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title(f'Issues per Day for {repo_name}')

    # Formatting x-axis to show dates
    plt.gca().xaxis_date()
    plt.gcf().autofmt_xdate()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'issues_' + repo_name + '.png'
    upload_blob_from_memory(buf, filename)

    buf.close()

    return bucketurl + filename


@app.route('/api/plot/issues/month', methods=['GET'])
def get_data_issues_month():
    # Extract unique months dynamically
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/issues/month?repo=' + repo)
    data = response_API.text
    jsondata = json.loads(data)
    all_months = set()
    for repo_data in jsondata.values():
        all_months.update(repo_data.keys())
    months = sorted(all_months)

    # Set the reduced bar width
    bar_width = 0.2  # Adjust this value to your preference

    # Create a figure with a smaller size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize the position of the ind for the bar
    repositories = list(jsondata.keys())
    index = np.arange(len(repositories))

    # Loop over the months and create a bar for each month
    for i, month in enumerate(months):
        monthly_data = [repo_data.get(month, 0) for repo_data in jsondata.values()]
        ax.bar(index + i * bar_width, monthly_data, bar_width, label=month)

    # Set the x-axis tick labels to the names of the repositories
    ax.set_xticks(index + bar_width * len(months) / 2)
    ax.set_xticklabels(repositories, rotation='vertical')

    # Set the title and labels for the axes
    ax.set_xlabel('Repository')
    ax.set_ylabel('Number of Issues')
    ax.set_title('Number of Issues by Repository and Month')
    ax.legend()

    # Show the plot
    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'issues_month_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename


@app.route('/api/plot/issues/week', methods=['GET'])
def get_data_issues_week():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/issues/week?repo=' + repo)
    data = response_API.text
    jsondata = json.loads(data)
    # Extract dates and categories
    all_weeks = sorted({week for repo_data in jsondata.values() for week in repo_data})

    fig, ax = plt.subplots()

    bar_width = 0.2
    index = np.arange(len(jsondata))

    # Loop over the sorted weeks
    for i, week in enumerate(all_weeks):
        weekly_data = [repo_data.get(week, 0) for repo_data in jsondata.values()]
        ax.bar(index + i * bar_width, weekly_data, bar_width, label=week)

    ax.set_xticks(index + bar_width * len(all_weeks) / 2)
    ax.set_xticklabels(jsondata.keys(), rotation='vertical')

    ax.set_xlabel('Repository')
    ax.set_ylabel('Number of Issues Closed')
    ax.set_title('Number of Issues Closed by Repository and Week')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'issues_week_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename


@app.route('/api/plot/issues/created-closed', methods=['GET'])
def get_data_issues_created_closed():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/issues/created-closed?repo=' + repo)
    data = response_API.text
    data = json.loads(data)
    # Extract dates and categories
    dates = sorted(set(date for repo_data in data.values() for date in repo_data.keys()))
    categories = ["created", "closed"]

    # Create a figure with a size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Initialize the position of the bars
    index = np.arange(len(dates))

    # Loop through categories and create stacked bars
    for category in categories:
        bottom = np.zeros(len(dates))
        for repo_data in data.values():
            values = [repo_data.get(date, {}).get(category, 0) for date in dates]
            ax.bar(index, values, label=f"{category.capitalize()} Issues", bottom=bottom)
            bottom += values

    # Set the x-axis tick labels to the dates
    ax.set_xticks(index)
    ax.set_xticklabels(dates, rotation='vertical')

    # Set the title and labels for the axes
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Issues')
    ax.set_title('Stacked Bar Chart: Number of Created and Closed Issues by Date')
    ax.legend()

    # Show the plot
    fig.tight_layout()
    fig.tight_layout()
    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'issues_created-closed_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename


@app.route('/api/plot/fork', methods=['GET'])
def get_data_fork():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/fork-stars?repo=' + repo)
    data = response_API.text
    jsondata = json.loads(data)

    # Extract forks data
    forks_data = jsondata[repo]["forks_counts_last_two_months"]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert the date strings to datetime for sorting
    dates = sorted(forks_data.keys())
    forks_count = [forks_data[date] for date in dates]

    # Plot the data
    ax.bar(dates, forks_count, color='blue', alpha=0.7, label='Forks')

    # Set the title and labels for the axes
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Forks')
    ax.set_title('Number of Forks Over the Last Two Months')
    ax.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'fork_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename


@app.route('/api/plot/stars', methods=['GET'])
def get_data_stars():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/fork-stars?repo=' + repo)
    data = response_API.text
    jsondata = json.loads(data)

    # Extract forks data
    forks_data = jsondata[repo]["stargazers_counts_last_two_months"]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert the date strings to datetime for sorting
    dates = sorted(forks_data.keys())
    forks_count = [forks_data[date] for date in dates]

    # Plot the data
    ax.bar(dates, forks_count, color='red', alpha=0.7, label='Forks')

    # Set the title and labels for the axes
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Stars')
    ax.set_title('Number of Stars Over the Last Two Months')
    ax.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'stars_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename


@app.route('/api/lstm/issues/open', methods=['GET'])
def get_forecast_issues_open():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/issues/open?repo=' + repo)
    data = response_API.text
    data = json.loads(data)
    df = pd.DataFrame(data[repo])

    # Convert 'created_at' to datetime and extract the day of the week
    df['created_at'] = pd.to_datetime(df['created_at']).dt.date
    df['day_of_week'] = pd.to_datetime(df['created_at']).dt.dayofweek

    # Aggregate data by day of the week
    agg_data = df.groupby('day_of_week').size().reset_index(name='issue_count')
    # Convert to numpy array
    data = agg_data['issue_count'].values

    # Prepare data for LSTM
    # Since we're dealing with days of the week, let's use the last few weeks to predict the next week
    X, y = [], []
    n_past = 4  # Number of past weeks to consider
    n_future = 1  # Predicting the next week

    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i])
        y.append(data[i:i + n_future])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshaping for LSTM
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(n_past, 1)))
    model.add(Dense(units=n_future))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=100, batch_size=32)
    # Forecast
    forecast = model.predict(X)

    # Find the day with the maximum number of predicted issues
    predicted_max_day = np.argmax(np.sum(forecast, axis=0))

    # Convert day index to day name
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    predicted_max_day_name = days_of_week[predicted_max_day]

    return predicted_max_day_name


@app.route('/api/lstm/issues/closed', methods=['GET'])
def get_forecast_issues_closed():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/issues/closed?repo=' + repo)
    data = response_API.text
    dataclosed = json.loads(data)
    # Convert to DataFrame
    dfclosed = pd.DataFrame(dataclosed[repo])

    # Convert 'created_at' to datetime and extract the day of the week
    dfclosed['closed_at'] = pd.to_datetime(dfclosed['closed_at']).dt.date
    dfclosed['day_of_week'] = pd.to_datetime(dfclosed['closed_at']).dt.dayofweek

    # Aggregate data by day of the week
    agg_dataclosed = dfclosed.groupby('day_of_week').size().reset_index(name='issue_count')
    closeddata = agg_dataclosed['issue_count'].values

    # Data preparation for LSTM
    Xclosed, yclosed = [], []
    n_past = 4  # Using 4 weeks of data to predict
    n_future = 1  # Predicting one week ahead

    for i in range(n_past, len(closeddata) - n_future + 1):
        Xclosed.append(closeddata[i - n_past:i])
        yclosed.append(closeddata[i:i + n_future])

    Xclosed, yclosed = np.array(Xclosed), np.array(yclosed)
    Xclosed = Xclosed.reshape((Xclosed.shape[0], Xclosed.shape[1], 1))  # Reshaping for LSTM

    # Model building
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(n_past, 1)))
    model.add(Dense(units=n_future))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Model training
    model.fit(Xclosed, yclosed, epochs=100, batch_size=32)

    # Forecasting
    forecast = model.predict(Xclosed)

    # Find the day with the maximum number of predicted issues
    predicted_max_day = np.argmax(np.sum(forecast, axis=0))
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    predicted_max_day_name = days_of_week[predicted_max_day]

    return predicted_max_day_name


@app.route('/api/lstm/issues/month', methods=['GET'])
def get_forecast_issues_month():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/issues/closed?repo=' + repo)
    data = response_API.text
    dataclosed = json.loads(data)
    # Convert to DataFrame
    df_closed = pd.DataFrame(dataclosed[repo])
    # Convert 'closed_at' to datetime and extract the month
    # Convert to DataFrame
    # Convert 'closed_at' to datetime and extract the month
    df_closed['closed_at'] = pd.to_datetime(df_closed['closed_at'])
    df_closed['month'] = df_closed['closed_at'].dt.month

    # Count issues per month
    monthly_issue_count = df_closed.groupby('month').size()

    # Resample to ensure all months are represented (including months with zero issues)
    monthly_issue_count = monthly_issue_count.reindex(range(1, 13), fill_value=0)
    # Prepare data for LSTM
    X_closed, y_closed = [], []
    n_past_months = 6  # Number of past months to consider
    n_future_months = 1  # Predicting the next month

    for i in range(n_past_months, len(monthly_issue_count) - n_future_months + 1):
        X_closed.append(monthly_issue_count[i - n_past_months:i])
        y_closed.append(monthly_issue_count[i:i + n_future_months])

    X_closed, y_closed = np.array(X_closed), np.array(y_closed)
    X_closed = X_closed.reshape((X_closed.shape[0], X_closed.shape[1], 1))  # Reshaping for LSTM
    # Build the LSTM model
    model_closed = Sequential()
    model_closed.add(LSTM(units=50, activation='relu', input_shape=(n_past_months, 1)))
    model_closed.add(Dense(units=n_future_months))
    model_closed.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model_closed.fit(X_closed, y_closed, epochs=100, batch_size=32)
    # Forecast
    forecast_closed = model_closed.predict(X_closed)

    # Find the month with the maximum number of predicted closed issues
    predicted_max_month = np.argmax(np.sum(forecast_closed, axis=0)) + 1  # Adding 1 as months are 1-indexed

    return str(predicted_max_month)

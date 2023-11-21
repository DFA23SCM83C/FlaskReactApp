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
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
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


@app.route('/api/lstm/plot/issues/created', methods=['GET'])
def get_forecast_plot_issues_created():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/issues/open?repo=' + repo)
    data = response_API.text
    dataissues = json.loads(data)
    # Convert to DataFrame and process dates
    dfissues = pd.DataFrame(dataissues[repo])
    dfissues['created_at'] = pd.to_datetime(dfissues['created_at'])
    dfissues.set_index('created_at', inplace=True)

    # Resample and count issues per day
    daily_issue_count = dfissues.resample('D').size()

    # Convert Series to DataFrame for consistent handling
    daily_issue_count_df = daily_issue_count.to_frame(name='count')

    # Prepare data for LSTM
    Xissues, yissues = [], []
    n_past = 7  # Number of past days to consider
    n_future = 1  # Predicting the next day

    for i in range(n_past, len(daily_issue_count_df) - n_future + 1):
        Xissues.append(daily_issue_count_df.iloc[i - n_past:i, 0])
        yissues.append(daily_issue_count_df.iloc[i, 0])

    Xissues, yissues = np.array(Xissues), np.array(yissues)
    Xissues = Xissues.reshape((Xissues.shape[0], n_past, 1))  # Reshaping for LSTM

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_past, 1)))
    model.add(Dense(n_future))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(Xissues, yissues, epochs=100, batch_size=32)
    n_future_days = 30  # Number of days to forecast
    last_known_date = daily_issue_count_df.index[-1]
    future_dates = [last_known_date + timedelta(days=x) for x in range(1, n_future_days + 1)]

    # Use the last n_past days from your data for the initial forecast
    last_known_data = np.array(daily_issue_count_df[-n_past:]).reshape(1, n_past, 1)

    # Iteratively predict each future day
    future_forecast = []
    for i in range(n_future_days):
        next_day_prediction = model.predict(last_known_data)
        future_forecast.append(next_day_prediction.flatten()[0])
        # Update last_known_data with the new prediction
        last_known_data = np.roll(last_known_data, -1, axis=1)
        last_known_data[0, -1, 0] = next_day_prediction

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(daily_issue_count_df.index, daily_issue_count_df['count'], label='Actual Issue Count')
    plt.plot(future_dates, future_forecast, label='Forecasted Issue Count', color='orange')
    plt.title('Forecast of Issue Creation')
    plt.xlabel('Date')
    plt.ylabel('Number of Issues')
    plt.xticks(rotation=45)
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'lstm_issues_created_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename


@app.route('/api/lstm/plot/issues/closed', methods=['GET'])
def get_forecast_plot_issues_closed():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/issues/closed?repo=' + repo)
    data = response_API.text
    dataclosed = json.loads(data)
    dfclosed = pd.DataFrame(dataclosed[repo])
    dfclosed['closed_at'] = pd.to_datetime(dfclosed['closed_at'])
    dfclosed.set_index('closed_at', inplace=True)

    # Resample and count closed issues per day
    daily_closed_count = dfclosed.resample('D').size()

    # Convert Series to DataFrame for consistent handling
    daily_closed_count_df = daily_closed_count.to_frame(name='count')
    # Prepare data for LSTM
    Xclosed, yclosed = [], []
    n_past = 7  # Number of past days to consider
    n_future = 1  # Predicting the next day

    for i in range(n_past, len(daily_closed_count_df) - n_future + 1):
        Xclosed.append(daily_closed_count_df.iloc[i - n_past:i, 0])
        yclosed.append(daily_closed_count_df.iloc[i, 0])

    Xclosed, yclosed = np.array(Xclosed), np.array(yclosed)
    Xclosed = Xclosed.reshape((Xclosed.shape[0], n_past, 1))  # Reshaping for LSTM

    # Build the LSTM model
    closed_model = Sequential()
    closed_model.add(LSTM(50, activation='relu', input_shape=(n_past, 1)))
    closed_model.add(Dense(n_future))
    closed_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    closed_model.fit(Xclosed, yclosed, epochs=100, batch_size=32)

    # Forecast
    n_future_days = 30  # Number of days to forecast for closed issues
    last_known_date_closed = daily_closed_count_df.index[-1]
    future_dates_closed = [last_known_date_closed + timedelta(days=x) for x in range(1, n_future_days + 1)]

    last_known_data_closed = np.array(daily_closed_count_df[-n_past:]).reshape(1, n_past, 1)

    future_forecast_closed = []
    for i in range(n_future_days):
        next_day_prediction_closed = closed_model.predict(last_known_data_closed)
        future_forecast_closed.append(next_day_prediction_closed.flatten()[0])
        last_known_data_closed = np.roll(last_known_data_closed, -1, axis=1)
        last_known_data_closed[0, -1, 0] = next_day_prediction_closed
    plt.figure(figsize=(15, 7))
    plt.plot(daily_closed_count_df.index, daily_closed_count_df['count'], label='Actual Closed Issue Count')
    plt.plot(future_dates_closed, future_forecast_closed, label='Forecasted Closed Issue Count', color='orange')
    plt.title('Forecast of Issue Closure')
    plt.xlabel('Date')
    plt.ylabel('Number of Closed Issues')
    plt.xticks(rotation=45)
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'lstm_issues_closed_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename


@app.route('/api/lstm/plot/commits', methods=['GET'])
def get_forecast_plot_commits():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/commits?repo=' + repo)
    data = response_API.text
    datacommit = json.loads(data)

    df = pd.DataFrame()
    for repo, commits in datacommit.items():
        temp_df = pd.DataFrame(commits)

        # Check if 'date' key is in the DataFrame columns
        if 'date' in temp_df.columns:
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            df = pd.concat([df, temp_df])
        else:
            print(f"'date' key not found in the commits for repository {repo}")

    # Group by date and count the number of commits
    df = df.groupby('date')['sha'].count().reset_index(name='count')

    # Resample to ensure all days are represented
    df.set_index('date', inplace=True)
    df = df.resample('D').asfreq().fillna(0)

    # Prepare data for LSTM
    n_past = 7
    n_future = 1
    X, y = [], []
    for i in range(n_past, len(df) - n_future + 1):
        X.append(df.iloc[i - n_past:i, 0])
        y.append(df.iloc[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], n_past, 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_past, 1)))
    model.add(Dense(n_future))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=100, batch_size=32)

    # Forecast future commits
    n_future_days = 30
    last_known_data = X[-1:]
    future_forecast = []

    for _ in range(n_future_days):
        next_day_prediction = model.predict(last_known_data)
        future_forecast.append(next_day_prediction.flatten()[0])
        last_known_data = np.roll(last_known_data, -1, axis=1)
        last_known_data[0, -1, 0] = next_day_prediction

    # Create dates for the forecast
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_future_days + 1)]

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['count'], label='Actual Commit Count')
    plt.plot(future_dates, future_forecast, label='Forecasted Commit Count', color='orange')
    plt.title('Forecast of Commit Activity')
    plt.xlabel('Date')
    plt.ylabel('Number of Commits')
    plt.xticks(rotation=45)
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'lstm_commits_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename


@app.route('/api/lstm/plot/pull', methods=['GET'])
def get_forecast_plot_pull():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/pull?repo=' + repo)
    data = response_API.text
    datapull = json.loads(data)
    # Convert data into DataFrame
    df = pd.DataFrame(datapull[repo])

    # Convert 'created_at' to datetime and count PRs per day
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.groupby('created_at').count().reset_index()

    # Resample to ensure all days are represented
    df.set_index('created_at', inplace=True)
    df = df.resample('D').asfreq().fillna(0)

    # Prepare data for LSTM
    n_past = 7
    n_future = 1
    X, y = [], []
    for i in range(n_past, len(df) - n_future + 1):
        X.append(df.iloc[i - n_past:i, 0])
        y.append(df.iloc[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], n_past, 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_past, 1)))
    model.add(Dense(n_future))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=100, batch_size=32)

    # Forecast future PRs
    n_future_days = 30
    last_known_data = X[-1:]
    future_forecast = []

    for _ in range(n_future_days):
        next_day_prediction = model.predict(last_known_data)
        future_forecast.append(next_day_prediction.flatten()[0])
        last_known_data = np.roll(last_known_data, -1, axis=1)
        last_known_data[0, -1, 0] = next_day_prediction

    # Create dates for the forecast
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_future_days + 1)]

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['pr_number'], label='Actual Pull Count')
    plt.plot(future_dates, future_forecast, label='Forecasted Pull Count', color='orange')
    plt.title('Forecast of Pull Activity')
    plt.xlabel('Date')
    plt.ylabel('Number of pulls')
    plt.xticks(rotation=45)
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'lstm_pull_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename

@app.route('/api/lstm/plot/contributor', methods=['GET'])
def get_forecast_plot_contributor():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/contributors?repo=' + repo)
    data = response_API.text
    datacontributors = json.loads(data)
    # Step 1: Preprocess the Data
    dates = []
    for repo, contributors in datacontributors.items():
        for contributor in contributors:
            dates.extend(contributor['dates'])

    # Convert to DataFrame
    df = pd.DataFrame({'date': dates})
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby('date').size().reset_index(name='contributions')

    # Ensure continuous dates and fill missing values
    df = df.set_index('date').asfreq('D').fillna(0)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)

    # Step 2: Prepare Training Data
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 5
    X, Y = create_dataset(scaled_data, look_back)

    # Reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Step 3: Build and Train LSTM Model
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, epochs=100, batch_size=1, verbose=2)

    # Step 4: Forecast Future Contributions
    predictions = []
    current_batch = X[-1:]

    for i in range(30):  # Forecast next 30 days
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    predicted_contributions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Convert predictions to a suitable format for plotting
    future_dates = pd.date_range(start=df.index[-1], periods=30)
    future_df = pd.DataFrame(data=predicted_contributions, index=future_dates, columns=['Predictions'])

    # Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['contributions'], label='Actual Contributions')
    plt.plot(future_df.index, future_df['Predictions'], label='Predicted Contributions', color='orange')
    plt.title('Contributions Forecast Using LSTM')
    plt.xlabel('Date')
    plt.ylabel('Number of Contributions')
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'lstm_contributor_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename

@app.route('/api/lstm/plot/release', methods=['GET'])
def get_forecast_plot_release():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/release?repo=' + repo)
    data = response_API.text
    datarelease = json.loads(data)
    # Extract release dates
    release_dates = [entry['release_date'] for entry in datarelease[repo]]

    # Convert to DataFrame
    df = pd.DataFrame({'release_date': release_dates})
    df['release_date'] = pd.to_datetime(df['release_date'])
    df = df.groupby('release_date').size().reset_index(name='releases')

    # Resample to daily frequency, filling days without releases with 0
    df = df.set_index('release_date').asfreq('D').fillna(0)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 5
    X, Y = create_dataset(scaled_data, look_back)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, epochs=100, batch_size=1, verbose=2)
    predictions = []
    current_batch = X[-1:]

    for i in range(30):  # Forecast next 30 days
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    predicted_releases = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
    future_df = pd.DataFrame(data=predicted_releases, index=future_dates, columns=['Predictions'])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['releases'], label='Actual Releases')
    plt.plot(future_df.index, future_df['Predictions'], label='Predicted Releases', color='orange')
    plt.title('Release Forecast Using LSTM')
    plt.xlabel('Date')
    plt.ylabel('Number of Releases')
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'lstm_release_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename



@app.route('/api/prophet/issues/created', methods=['GET'])
def get_forecast_prohphet_issues_created():
    repo = request.args.get('repo')
    appurl = 'https://backendflask-e64bz5ofya-uc.a.run.app'
    response_API = requests.get(appurl + '/api/date/issues/open?repo=' + repo)
    data = response_API.text
    data = json.loads(data)
    df = pd.DataFrame(data[repo])

    # Convert 'created_at' to datetime
    df['ds'] = pd.to_datetime(df['created_at']).dt.date

    # Aggregate data by day
    df = df.groupby('ds').size().reset_index(name='y')

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Create a future DataFrame for prediction
    future = model.make_future_dataframe(periods=30)  # Predicting the next 30 days

    # Make predictions
    forecast = model.predict(future)

    # Aggregate predictions by day of the week
    forecast['day_of_week'] = forecast['ds'].dt.dayofweek
    weekly_forecast = forecast.groupby('day_of_week')['yhat'].sum().reset_index()

    # Find the day with the maximum predicted issues
    predicted_max_day = weekly_forecast['day_of_week'][weekly_forecast['yhat'].idxmax()]

    # Convert day index to day name
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    predicted_max_day_name = days_of_week[predicted_max_day]

    return predicted_max_day_name


@app.route('/api/prophet/issues/closed', methods=['GET'])
def get_forecast_prohphet_issues_closed():
    repo = request.args.get('repo')
    appurl = 'https://backendflask-e64bz5ofya-uc.a.run.app'
    response_API = requests.get(appurl + '/api/date/issues/closed?repo=' + repo)
    data = response_API.text
    data = json.loads(data)
    df = pd.DataFrame(data[repo])

    # Convert 'created_at' to datetime
    df['ds'] = pd.to_datetime(df['closed_at']).dt.date

    # Aggregate data by day
    df = df.groupby('ds').size().reset_index(name='y')

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Create a future DataFrame for prediction
    future = model.make_future_dataframe(periods=30)  # Predicting the next 30 days

    # Make predictions
    forecast = model.predict(future)

    # Aggregate predictions by day of the week
    forecast['day_of_week'] = forecast['ds'].dt.dayofweek
    weekly_forecast = forecast.groupby('day_of_week')['yhat'].sum().reset_index()

    # Find the day with the maximum predicted issues
    predicted_max_day = weekly_forecast['day_of_week'][weekly_forecast['yhat'].idxmax()]

    # Convert day index to day name
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    predicted_max_day_name = days_of_week[predicted_max_day]

    return predicted_max_day_name


@app.route('/api/prophet/issues/month', methods=['GET'])
def get_forecast_prohphet_issues_month():
    repo =request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/issues/closed?repo=' + repo)
    data = response_API.text
    dataclosed = json.loads(data)
    df_closed = pd.DataFrame(dataclosed[repo])

    # Convert 'closed_at' to datetime and extract the month
    df_closed['closed_at'] = pd.to_datetime(df_closed['closed_at'])
    df_closed['month'] = df_closed['closed_at'].dt.to_period('M')

    # Count issues per month and reset index
    monthly_issue_count = df_closed.groupby('month').size().reset_index(name='y')
    monthly_issue_count['ds'] = monthly_issue_count['month'].dt.to_timestamp()

    # Instantiate and fit the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(monthly_issue_count[['ds', 'y']])

    # Create future DataFrame for the next 12 months
    future = model.make_future_dataframe(periods=1, freq='M')

    # Make predictions
    forecast = model.predict(future)

    # Aggregate predictions by month
    forecast['month'] = forecast['ds'].dt.to_period('M')
    monthly_forecast = forecast.groupby('month')['yhat'].sum().reset_index()

    # Find the month with the maximum predicted issues
    predicted_max_month = monthly_forecast['month'][monthly_forecast['yhat'].idxmax()]

    return str(predicted_max_month)


@app.route('/api/prophet/plot/issues/created', methods=['GET'])
def get_forecast_plot_issues_created_prophet():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/issues/open?repo=' + repo)
    dataissues = json.loads(response_API.text)

    # Convert to DataFrame and process dates
    df_issues = pd.DataFrame(dataissues[repo])
    df_issues['created_at'] = pd.to_datetime(df_issues['created_at']).dt.date

    # Prepare data for Prophet
    df_prophet = df_issues.groupby('created_at').size().reset_index(name='y')
    df_prophet.columns = ['ds', 'y']

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Create future DataFrame for next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plotting
    fig = model.plot(forecast)
    plt.title(f'Forecast of Issue Creation for {repo}')
    plt.ylabel('Number of Issues')
    plt.xlabel('Date')

    # Save plot to memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'prophet_issues_created_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename

@app.route('/api/prophet/plot/issues/closed', methods=['GET'])
def get_forecast_plot_issues_closed_prophet():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/issues/closed?repo=' + repo)
    dataclosed = json.loads(response_API.text)
    df_closed = pd.DataFrame(dataclosed[repo])

    # Prepare data for Prophet
    df_closed['closed_at'] = pd.to_datetime(df_closed['closed_at']).dt.date
    df_prophet = df_closed.groupby('closed_at').size().reset_index(name='y')
    df_prophet.columns = ['ds', 'y']

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Create future DataFrame for next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plotting
    fig = model.plot(forecast)
    plt.title(f'Forecast of Issue Closure for {repo}')
    plt.ylabel('Number of Closed Issues')
    plt.xlabel('Date')

    # Save plot to memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'prophet_issues_closed_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename


@app.route('/api/prophet/plot/commits', methods=['GET'])
def get_forecast_plot_commits_prophet():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/commits?repo=' + repo)
    datacommit = json.loads(response_API.text)

    df = pd.DataFrame()
    for repo, commits in datacommit.items():
        temp_df = pd.DataFrame(commits)
        if 'date' in temp_df.columns:
            temp_df['date'] = pd.to_datetime(temp_df['date'])
            df = pd.concat([df, temp_df])
        else:
            continue

    # Prepare data for Prophet
    df = df.groupby('date').size().reset_index(name='y')
    df.columns = ['ds', 'y']

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Create future DataFrame for next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plotting
    fig = model.plot(forecast)
    plt.title(f'Forecast of Commit Activity for {repo}')
    plt.ylabel('Number of Commits')
    plt.xlabel('Date')

    # Save plot to memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'prophet_commits_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename


@app.route('/api/prophet/plot/pull', methods=['GET'])
def get_forecast_plot_pull_prophet():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/pull?repo=' + repo)
    datapull = json.loads(response_API.text)

    # Convert data into DataFrame
    df = pd.DataFrame(datapull[repo])

    # Prepare data for Prophet
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.groupby('created_at').size().reset_index(name='y')
    df.columns = ['ds', 'y']

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Create future DataFrame for next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Plotting
    fig = model.plot(forecast)
    plt.title(f'Forecast of Pull Request Activity for {repo}')
    plt.ylabel('Number of Pull Requests')
    plt.xlabel('Date')

    # Save plot to memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'prophet_pull_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename


@app.route('/api/prophet/plot/contributor', methods=['GET'])
def get_forecast_plot_contributor_prophet():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/contributors?repo=' + repo)
    datacontributors = json.loads(response_API.text)

    # Step 1: Preprocess the Data
    dates = []
    for contributors in datacontributors.values():
        for contributor in contributors:
            dates.extend(contributor['dates'])

    # Convert to DataFrame
    df = pd.DataFrame({'ds': dates})
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.groupby('ds').size().reset_index(name='y')

    # Create and fit the Prophet model
    model = Prophet()
    model.fit(df)

    # Step 2: Make Future Predictions
    future_dates = model.make_future_dataframe(periods=30)
    forecast = model.predict(future_dates)

    # Step 3: Plot the Forecast
    fig = model.plot(forecast)
    plt.title('Contributions Forecast Using Prophet')
    plt.ylabel('Number of Contributions')
    plt.xlabel('Date')

    # Save plot to memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'prophet_contributor_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename

@app.route('/api/prophet/plot/release', methods=['GET'])
def get_forecast_plot_release_prophet():
    repo = request.args.get('repo')
    response_API = requests.get(appurl + '/api/date/release?repo=' + repo)
    datarelease = json.loads(response_API.text)

    # Prepare the Data
    release_dates = [entry['release_date'] for entry in datarelease[repo]]
    df = pd.DataFrame({'ds': release_dates})
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.groupby('ds').size().reset_index(name='y')

    # Create and Fit the Prophet Model
    model = Prophet()
    model.fit(df)

    # Make Future Predictions
    future_dates = model.make_future_dataframe(periods=30)
    forecast = model.predict(future_dates)

    # Plot the Forecast
    fig = model.plot(forecast)
    plt.title('Release Forecast Using Prophet')
    plt.ylabel('Number of Releases')
    plt.xlabel('Date')

    # Save plot to memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    filename = 'prophet_release_' + repo + '.png'
    upload_blob_from_memory(buf, filename)
    buf.close()

    return bucketurl + filename

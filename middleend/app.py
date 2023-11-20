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



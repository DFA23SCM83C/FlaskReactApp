from datetime import datetime, timedelta
import json
import pytz
from flask import Flask, render_template,jsonify
from flask_cors import CORS
import time
from collections import  defaultdict
import github3


app = Flask(__name__)
# app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30000)

GITHUB_TOKEN = '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'
gh = github3.login(token=GITHUB_TOKEN)
two_months_ago = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
REPOSITORIES = [
    'openai/openai-cookbook',
    'openai/openai-python',
    'openai/openai-quickstart-python',
    'milvus-io/pymilvus',
    'SeleniumHQ/selenium',
    'golang/go',
    'google/go-github',
    'angular/material',
    'angular/angular-cli',
    'sebholstein/angular-google-maps',
    'd3/d3',
    'facebook/react',
    'tensorflow/tensorflow',
    'keras-team/keras',
    'pallets/flask'
]
def get_rate_limit():
    rate_limit = gh.rate_limit()
    core = rate_limit['resources']['core']
    remaining = core['remaining']
    reset_time = datetime.fromtimestamp(core['reset'])
    return remaining, reset_time

# stars and forks
def get_stargazers_and_forks(repo_name):
    utc = pytz.UTC
    two_months_ago = datetime.now() - timedelta(days=60)
    two_months_ago = utc.localize(two_months_ago)  # Make 'two_months_ago' timezone-aware

    stargazers = 0
    forks = 0

    owner, name = repo_name.split('/')
    repository = gh.repository(owner, name)

    for event in repository.events():
        if event.created_at < two_months_ago:
            break
        if event.type == 'WatchEvent':  # This is a stargazing event
            stargazers += 1

    for event in repository.events():
        if event.created_at < two_months_ago:
            break
        if event.type == 'ForkEvent':  # This is a fork event
            forks += 1

    return {
        'stargazers_last_two_months': stargazers,
        'forks_last_two_months': forks
    }
# issues
def get_issue_count(repo_name):
    issue_count = 0
    remaining, reset_time = get_rate_limit()
    if remaining == 0:
        sleep_time = (reset_time - datetime.utcnow()).total_seconds() + 10
        print(f"Rate limit exceeded, sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)
    
    try:
        for _ in gh.search_issues(f'type:issue repo:{repo_name} created:>={two_months_ago}'):
            issue_count += 1
    except github3.exceptions.ForbiddenError as e:
        print('Rate limit exceeded, waiting to retry...')
        reset_time = e.response.headers['X-RateLimit-Reset']
        sleep_time = (datetime.utcfromtimestamp(int(reset_time)) - datetime.utcnow()).total_seconds() + 10
        print(f"Sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)
        return get_issue_count(repo_name)  # Recursively retry fetching issue count after sleep
    except github3.exceptions.GitHubError as e:
        print(f'Error fetching issues for repository {repo_name}: {e}')
    return issue_count
# issues per month
def get_issues_per_month(repo_name):
    # Dictionary to hold issue counts per month
    issues_per_month = defaultdict(int)
    issue_count = 0
    remaining, reset_time = get_rate_limit()
    if remaining == 0:
        sleep_time = (reset_time - datetime.utcnow()).total_seconds() + 10
        print(f"Rate limit exceeded, sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)
    
    try:
    # Fetch issues created in the last two months
       for issue in gh.search_issues(f'type:issue repo:{repo_name} created:>={two_months_ago}'):
        issue_data = issue.as_json()
        issue_data = json.loads(issue_data)
        created_at = datetime.strptime(issue_data['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        
        # Group by month and year, increment count
        month_year = created_at.strftime('%Y-%m')
        issues_per_month[month_year] += 1
    except github3.exceptions.ForbiddenError as e:
        print('Rate limit exceeded, waiting to retry...')
        # Parse the reset time from the response and wait until the reset time.
        reset_time = e.response.headers['X-RateLimit-Reset']
        sleep_time = (datetime.utcfromtimestamp(int(reset_time)) - datetime.utcnow()).total_seconds() + 10
        print(f"Sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)
        return get_issues_per_month(repo_name)  # Recursively retry fetching issues after sleep
    except github3.exceptions.GitHubError as e:
        print(f'Error fetching issues for repository {repo_name}: {e}')    

    return issues_per_month
# issues closed per week
def get_closed_issues_per_week(repo_name):
    # Calculate the date two months ago
    two_months_ago = (datetime.now() - timedelta(weeks=8)).isoformat()

    # Dictionary to hold issue counts per week
    issues_per_week = defaultdict(int)

    remaining, reset_time = get_rate_limit()
    if remaining == 0:
        sleep_time = (reset_time - datetime.utcnow()).total_seconds() + 10
        print(f"Rate limit exceeded, sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)

    try:
        # Fetch closed issues from the last two months
        for issue in gh.search_issues(f'type:issue repo:{repo_name} state:closed closed:>={two_months_ago}'):
            issue_data = issue.as_json()
            issue_data = json.loads(issue_data)
            closed_at = datetime.strptime(issue_data['closed_at'], '%Y-%m-%dT%H:%M:%SZ')

            # Group by week and increment count
            week_year = closed_at.strftime('%Y-%U')
            issues_per_week[week_year] += 1
    except github3.exceptions.ForbiddenError as e:
        print('Rate limit exceeded, waiting to retry...')
        # Parse the reset time from the response and wait until the reset time.
        reset_time = e.response.headers['X-RateLimit-Reset']
        sleep_time = (datetime.utcfromtimestamp(int(reset_time)) - datetime.utcnow()).total_seconds() + 10
        print(f"Sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)
        return get_closed_issues_per_week(repo_name)  # Recursively retry fetching issues after sleep
    except github3.exceptions.GitHubError as e:
        print(f'Error fetching issues for repository {repo_name}: {e}')
    
    return issues_per_week
#issues created and closed per repo
def get_issues_per_repo(repo_list):
    issues_per_repo = defaultdict(lambda: {'created': 0, 'closed': 0})

    for repo_name in repo_list:
        try:
            # Fetch issues created and closed in the last two months
            for issue in gh.search_issues(f'type:issue repo:{repo_name} created:>={two_months_ago}'):
                issues_per_repo[repo_name]['created'] += 1
                if issue.closed_at and issue.closed_at > two_months_ago:
                    issues_per_repo[repo_name]['closed'] += 1

            # Implement a small delay to avoid hitting rate limit
            time.sleep(10)  # Adjust the sleep duration as needed

        except github3.exceptions.ForbiddenError as e:
            print('Rate limit exceeded, waiting to retry...')
            sleep_time = 60  # Wait for 1 minute, adjust as needed
            time.sleep(sleep_time)
            return get_issues_per_repo(repo_list)  # Recursively retry fetching issues after sleep
        except github3.exceptions.GitHubError as e:
            print(f'Error fetching issues for repository {repo_name}: {e}')
            continue  # Proceed with the next repository

    return issues_per_repo


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/api/issues')
def get_data_issues_count():
    all_issue_counts = {repo: get_issue_count(repo) for repo in REPOSITORIES}
    return all_issue_counts

@app.route('/api/issues/month')
def get_data_issues_month():
    all_repo_issues_per_month = {repo: get_issues_per_month(repo) for repo in REPOSITORIES}
    issues_per_month_json = json.dumps(all_repo_issues_per_month)

    return issues_per_month_json

@app.route('/api/issues/week')
def get_data_issues_week():
    issues_per_week = {repo: get_closed_issues_per_week(repo) for repo in REPOSITORIES}
    issues_per_week_json = json.dumps(issues_per_week)

    return issues_per_week_json

@app.route('/api/fork-stars')
def get_data_fork_stars():
    all_datarepo = {repo: get_stargazers_and_forks(repo) for repo in REPOSITORIES}
 
    return all_datarepo

@app.route('/api/issues/created-closed')
def get_data_issues_created_closed():
    all_issues_per_repo =  get_issues_per_repo(REPOSITORIES)
    all_issues_per_repo_json = json.dumps(all_issues_per_repo)
    return all_issues_per_repo_json


@app.route('/api/data')
def get_data():
    # You would typically fetch data from a database or another service here
    data = {"key": "value"}
    return jsonify(data)






# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def serve(path):
#     if path != "" and os.path.exists(app.static_folder + '/' + path):
#         return send_from_directory(app.static_folder, path)
#     else:
#         return send_from_directory(app.static_folder, 'index.html')

# if __name__ == '__main__':
#     app.run()





# from flask import Flask,Response,jsonify, render_template ,logging,request
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# #run server
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=80)

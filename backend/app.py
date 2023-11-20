from datetime import datetime, timedelta, timezone
import json
import pytz
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import time
from collections import defaultdict
import github3
import requests
import os

app = Flask(__name__)
# app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30000)

GITHUB_TOKEN = os.environ['GITHUB_TOKEN']

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
    url = 'https://api.github.com/rate_limit'
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    remaining = data['rate']['remaining']
    reset_time = datetime.utcfromtimestamp(data['rate']['reset'])

    return remaining, reset_time


# stars and forks
def get_stargazers_and_forks(repo_name):
    utc = pytz.UTC
    two_months_ago = datetime.now() - timedelta(days=60)
    two_months_ago = utc.localize(two_months_ago)  # Make 'two_months_ago' timezone-aware

    stargazers_counts = {}
    forks_counts = {}

    owner, name = repo_name.split('/')
    repository = gh.repository(owner, name)

    for event in repository.events():
        if event.created_at < two_months_ago:
            break
        if event.type == 'WatchEvent':  # This is a stargazing event
            date_key = event.created_at.date()
            date_key_str = date_key.strftime("%Y-%m-%d")
            stargazers_counts[date_key_str] = stargazers_counts.get(date_key_str, 0) + 1

    for event in repository.events():
        if event.created_at < two_months_ago:
            break
        if event.type == 'ForkEvent':  # This is a fork event
            date_key = event.created_at.date()
            date_key_str = date_key.strftime("%Y-%m-%d")
            forks_counts[date_key_str] = forks_counts.get(date_key_str, 0) + 1

    result = {
        'stargazers_counts_last_two_months': stargazers_counts,
        'forks_counts_last_two_months': forks_counts
    }

    return result


# issues

def get_issue_count(repo_name):
    issue_counts = {}

    two_months_ago = datetime.utcnow() - timedelta(days=60)

    remaining, reset_time = get_rate_limit()
    if remaining == 0:
        sleep_time = (reset_time - datetime.utcnow()).total_seconds() + 10
        print(f"Rate limit exceeded, sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)

    try:
        # Fetch issues created in the last two months
        url = f'https://api.github.com/search/issues?q=type:issue repo:{repo_name} created:>={two_months_ago.strftime("%Y-%m-%dT%H:%M:%SZ")}'
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json',
        }

        response = requests.get(url, headers=headers)
        data = response.json()

        if response.status_code == 200:
            for issue in data.get('items', []):
                created_at = datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ').date()
                created_at_str = created_at.strftime("%Y-%m-%d")
                issue_counts[created_at_str] = issue_counts.get(created_at_str, 0) + 1
        else:
            print(f'Error fetching issues for repository {repo_name}: {response.content}')

    except requests.exceptions.RequestException as e:
        print(f'Error fetching issues for repository {repo_name}: {e}')

    return issue_counts


# issues per month
def get_issues_per_month(repo_name):
    # Dictionary to hold issue counts per month
    issues_per_month = defaultdict(int)
    issue_count = 0
    two_months_ago = (datetime.utcnow() - timedelta(days=60)).strftime('%Y-%m-%dT%H:%M:%SZ')

    remaining, reset_time = get_rate_limit()
    if remaining == 0:
        sleep_time = (reset_time - datetime.utcnow()).total_seconds() + 10
        print(f"Rate limit exceeded, sleeping for {sleep_time} seconds.")
        time.sleep(sleep_time)

    try:
        # Fetch issues created in the last two months
        url = f'https://api.github.com/search/issues?q=type:issue repo:{repo_name} created:>={two_months_ago}'
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json',
        }

        response = requests.get(url, headers=headers)
        data = response.json()

        if response.status_code == 200:
            for issue in data['items']:
                created_at = datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                month_year = created_at.strftime('%Y-%m')
                issues_per_month[month_year] += 1
        else:
            print(f'Error fetching issues for repository {repo_name}: {response.content}')

    except requests.exceptions.RequestException as e:
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


# issues created and closed per repo
def get_issues_per_repo(repo_name):
    issues_per_repo_over_time = defaultdict(lambda: {'created': 0, 'closed': 0})

    two_months_ago = (datetime.now() - timedelta(days=60)).isoformat()

    try:
        # Fetch issues created and closed in the last two months
        url = f'https://api.github.com/search/issues?q=type:issue repo:{repo_name} created:>={two_months_ago}'
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json',
        }

        response = requests.get(url, headers=headers)
        data = response.json()

        if response.status_code == 200:
            for issue in data['items']:
                created_at_date = datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ').date()
                closed_at_date = None

                if issue['closed_at']:
                    closed_at_date = datetime.strptime(issue['closed_at'], '%Y-%m-%dT%H:%M:%SZ').date()

                created_at_str = created_at_date.strftime("%Y-%m-%d")
                closed_at_str = closed_at_date.strftime("%Y-%m-%d") if closed_at_date else None

                issues_per_repo_over_time[created_at_str]['created'] += 1

                if closed_at_str and closed_at_str >= created_at_str:
                    issues_per_repo_over_time[closed_at_str]['closed'] += 1
        else:
            print(f'Error fetching issues for repository {repo_name}: {response.content}')

    except requests.exceptions.RequestException as e:
        print(f'Error fetching issues for repository {repo_name}: {e}')

    return dict(issues_per_repo_over_time)


def fetch_issues_for_repo(repo_name):
    two_months_ago = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    issues_data = []

    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }

    url = f'https://api.github.com/repos/{repo_name}/issues'
    params = {
        'state': 'all',
        'since': two_months_ago,
        'per_page': 100,
    }

    while url:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        if response.status_code == 200:
            for issue in data:
                issues_data.append({
                    'issue_number': issue['number'],
                    'created_at': issue['created_at'],
                })

            if 'next' in response.links:
                url = response.links['next']['url']
            else:
                url = None
        else:
            print(f'Error fetching issues for repository {repo_name}: {response.content}')
            break

    return issues_data


def fetch_closed_issues(repo_name):
    two_months_ago = datetime.now(timezone.utc) - timedelta(days=60)
    closed_issues_data = []

    # Split the repo_name into owner and repo
    owner, repo = repo_name.split('/')

    # Define the URL for the GitHub API
    url = f'https://api.github.com/repos/{owner}/{repo}/issues'

    # Define the parameters for the GET request
    params = {
        'state': 'closed',
        'since': two_months_ago.isoformat()
    }

    try:
        # Make the GET request
        response = requests.get(url, params=params)

        # Check if the request was successful
        response.raise_for_status()

        # Parse the response JSON
        issues = response.json()

        # Make two_months_ago aware of UTC timezone
        two_months_ago = two_months_ago.replace(tzinfo=timezone.utc)

        # Filter the issues that were closed in the last two months
        for issue in issues:
            if issue['closed_at']:
                closed_at = datetime.strptime(issue['closed_at'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                if closed_at >= two_months_ago:
                    closed_issues_data.append({
                        'issue_number': issue['number'],
                        'closed_at': closed_at.strftime('%Y-%m-%d')
                    })

    except requests.exceptions.HTTPError as e:
        print(f'Error fetching closed issues for repository {repo_name}: {e}')

    return closed_issues_data


def fetch_recent_pull_requests(repo_name):
    two_months_ago = datetime.now() - timedelta(days=60)
    recent_prs = []
    url = f"https://api.github.com/repos/{repo_name}/pulls?state=all"
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        pull_requests = response.json()

        for pr in pull_requests:
            pr_created_at = datetime.strptime(pr['created_at'], '%Y-%m-%dT%H:%M:%SZ')
            if pr_created_at >= two_months_ago:
                pr_data = {
                    'pr_number': pr['number'],
                    'created_at': pr_created_at.strftime('%Y-%m-%d'),
                }
                recent_prs.append(pr_data)

    except requests.exceptions.RequestException as e:
        print(f'Error fetching pull requests for repository {repo_name}: {e}')

    return recent_prs


def fetch_recent_commits(repo_name):
    two_months_ago = datetime.now() - timedelta(days=60)
    commits_summary = []
    url = f"https://api.github.com/repos/{repo_name}/commits"
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        commits = response.json()

        for commit in commits:
            commit_date = datetime.strptime(commit['commit']['author']['date'], '%Y-%m-%dT%H:%M:%SZ')
            if commit_date >= two_months_ago:
                commit_summary = {
                    'sha': commit['sha'],
                    'date': commit_date.strftime('%Y-%m-%d')
                }
                commits_summary.append(commit_summary)

    except requests.exceptions.RequestException as e:
        print(f'Error fetching commits for repository {repo_name}: {e}')

    return commits_summary


def get_branches_and_creation_date(repo_name):
    # Split the repo_name into owner and repo
    owner, repo = repo_name.split('/')

    # Define the URLs for the GitHub API
    branches_url = f'https://api.github.com/repos/{owner}/{repo}/branches'
    repo_url = f'https://api.github.com/repos/{owner}/{repo}'

    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }

    try:
        # Make the GET requests with authentication
        branches_response = requests.get(branches_url, headers=headers)
        repo_response = requests.get(repo_url, headers=headers)

        # Check if the requests were successful
        branches_response.raise_for_status()
        repo_response.raise_for_status()

        # Parse the response JSON
        branches = branches_response.json()
        repo_info = repo_response.json()

        # Get the creation date
        creation_date = datetime.strptime(repo_info['created_at'],'%Y-%m-%dT%H:%M:%SZ')

    except requests.exceptions.HTTPError as e:
        print(f'Error fetching branches and creation date for repository {repo_name}: {e}')
        return None

    # Create a list of dictionaries containing branch names and their creation dates
    branches_data = [{'branch': branch['name'], 'creation_date': creation_date.strftime('%Y-%m-%d')} for branch in branches]

    return branches_data


def get_contributors_and_dates(repo_name):
    # Split the repo_name into owner and repo
    owner, repo = repo_name.split('/')

    # Define the URL for the GitHub API to get contributors
    contributors_url = f'https://api.github.com/repos/{owner}/{repo}/contributors'

    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }

    contributors_data = []

    try:
        # Make the GET request for contributors
        contributors_response = requests.get(contributors_url, headers=headers)
        contributors_response.raise_for_status()
        contributors = contributors_response.json()

        for contributor in contributors:
            username = contributor['login']
            commits_url = f'https://api.github.com/repos/{owner}/{repo}/commits?author={username}'

            # Fetch the commits made by the contributor
            commits_response = requests.get(commits_url, headers=headers)
            commits_response.raise_for_status()
            commits = commits_response.json()

            # Extract dates from the commits
            dates = [datetime.strptime(commit['commit']['author']['date'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
                     for commit in commits]

            # Add to the contributors_data list
            contributors_data.append({'username': username, 'dates': dates})

    except requests.exceptions.HTTPError as e:
        print(f'Error fetching contributors and dates for repository {repo_name}: {e}')
        return None

    return contributors_data


def get_releases_and_dates(repo_name):
    # Split the repo_name into owner and repo
    owner, repo = repo_name.split('/')

    # Define the URL for the GitHub API to get releases
    releases_url = f'https://api.github.com/repos/{owner}/{repo}/releases'

    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }

    try:
        # Make the GET request with authentication
        releases_response = requests.get(releases_url, headers=headers)

        # Check if the request was successful
        releases_response.raise_for_status()

        # Parse the response JSON
        releases = releases_response.json()

        # Extract release names and their creation dates
        releases_data = [{'release_name': release['name'],
                          'release_date': datetime.strptime(release['created_at'], '%Y-%m-%dT%H:%M:%SZ').strftime(
                              '%Y-%m-%d')} for release in releases]

    except requests.exceptions.HTTPError as e:
        print(f'Error fetching releases and dates for repository {repo_name}: {e}')
        return None

    return releases_data


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/issues', methods=['GET'])
def get_data_issues_count():
    repo = request.args.get('repo')
    all_issue_counts = {repo: get_issue_count(repo)}
    return all_issue_counts


@app.route('/api/issues/month', methods=['GET'])
def get_data_issues_month():
    repo = request.args.get('repo')
    all_repo_issues_per_month = {repo: get_issues_per_month(repo)}
    issues_per_month_json = json.dumps(all_repo_issues_per_month)

    return issues_per_month_json


@app.route('/api/issues/week', methods=['GET'])
def get_data_issues_week():
    repo = request.args.get('repo')
    issues_per_week = {repo: get_closed_issues_per_week(repo)}
    issues_per_week_json = json.dumps(issues_per_week)
    return issues_per_week_json


@app.route('/api/fork-stars', methods=['GET'])
def get_data_fork_stars():
    repo = request.args.get('repo')
    all_datarepo = {repo: get_stargazers_and_forks(repo)}
    return all_datarepo


@app.route('/api/issues/created-closed', methods=['GET'])
def get_data_issues_created_closed():
    repo = request.args.get('repo')
    all_issues_per_repo = {repo: get_issues_per_repo(repo)}
    return all_issues_per_repo


@app.route('/api/date/issues/open', methods=['GET'])
def get_date_issues_open():
    repo = request.args.get('repo')
    # You would typically fetch data from a database or another service here
    issues_repository_data = {repo: fetch_issues_for_repo(repo)}
    return issues_repository_data


@app.route('/api/date/issues/closed', methods=['GET'])
def get_date_issues_closed():
    repo = request.args.get('repo')
    # You would typically fetch data from a database or another service here
    closed_issues_by_repo = {repo: fetch_closed_issues(repo)}
    return closed_issues_by_repo


@app.route('/api/date/pull', methods=['GET'])
def get_date_pull():
    repo = request.args.get('repo')
    # You would typically fetch data from a database or another service here
    recent_pull_requests = {repo: fetch_recent_pull_requests(repo)}
    return recent_pull_requests


@app.route('/api/date/commits', methods=['GET'])
def get_date_commits():
    repo = request.args.get('repo')
    # You would typically fetch data from a database or another service here
    recent_commits = {repo: fetch_recent_commits(repo)}
    return recent_commits


@app.route('/api/date/branch', methods=['GET'])
def get_date_branch():
    repo = request.args.get('repo')
    # You would typically fetch data from a database or another service here
    recent_branches = {repo: get_branches_and_creation_date(repo)}
    return recent_branches


@app.route('/api/date/contributors', methods=['GET'])
def get_date_contributors():
    repo = request.args.get('repo')
    # You would typically fetch data from a database or another service here
    recent_contributors_requests = {repo: get_contributors_and_dates(repo)}
    return recent_contributors_requests


@app.route('/api/date/release', methods=['GET'])
def get_date_release():
    repo = request.args.get('repo')
    # You would typically fetch data from a database or another service here
    recent_release_requests = {repo: get_releases_and_dates(repo)}
    return recent_release_requests


@app.route('/api/data')
def get_data():
    # You would typically fetch data from a database or another service here
    data = {"key": "value"}
    return jsonify(data)

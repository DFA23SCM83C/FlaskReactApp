import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import IssuesPage from './Page/Issues';
import IssuesMonthPage from "./Page/IssuesMonth";
import IssuesWeekPage from "./Page/IssuesWeek";
import IssuesCreatedClosedPage from "./Page/IssuesCreatedClosed";
import ForkPage from "./Page/Fork";
import StarsPage from "./Page/Stars";
import { useState } from 'react';
function App() {
    const REPOSITORIES = [
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
    const middleend = 'https://middleendflask-zjd2eijkya-uc.a.run.app'
    const textapi = {
        'openai/openai-cookbook': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'openai/openai-python': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'openai/openai-quickstart-python': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'milvus-io/pymilvus': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'SeleniumHQ/selenium': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'golang/go': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'google/go-github': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'angular/material': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'angular/angular-cli': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'sebholstein/angular-google-maps': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'SeleniumHQ/selenium': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'd3/d3': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'facebook/react': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'tensorflow/tensorflow': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'keras-team/keras': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
        'pallets/flask': [
            '/api/lstm/issues/open',
            '/api/lstm/issues/closed',
            '/api/lstm/issues/month',
            '/api/prophet/issues/created',
            '/api/prophet/issues/closed',
            '/api/prophet/issues/month',
            '/api/statsmodel/issues/open',
            '/api/statsmodel/issues/closed',
            '/api/statsmodel/issues/closed/month',
        ],
    }





    const repoEndpoints = {
        'openai/openai-cookbook': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'openai/openai-quickstart-python': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'openai/openai-python': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'milvus-io/pymilvus': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'SeleniumHQ/selenium': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'golang/go': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'google/go-github': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'angular/material': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'angular/angular-cli': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'sebholstein/angular-google-maps': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'd3/d3': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'facebook/react': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'tensorflow/tensorflow': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'keras-team/keras': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],
        'pallets/flask': [
            '/api/plot/issues',
            '/api/plot/issues/month',
            '/api/plot/issues/week',
            '/api/plot/issues/created-closed',
            '/api/plot/fork',
            '/api/plot/stars',
            '/api/lstm/plot/issues/created',
            '/api/lstm/plot/issues/closed',
            '/api/lstm/plot/commits',
            '/api/lstm/plot/pull',
            '/api/lstm/plot/contributor',
            '/api/lstm/plot/release',
            '/api/prophet/plot/issues/created',
            '/api/prophet/plot/issues/closed',
            '/api/prophet/plot/commits',
            '/api/prophet/plot/pull',
            '/api/prophet/plot/contributor',
            '/api/prophet/plot/release',
            '/api/statsmodel/plot/issues/created',
            '/api/statsmodel/plot/issues/closed',
            '/api/statsmodel/plot/commits',
            '/api/statsmodel/plot/pull',
            '/api/statsmodel/plot/contributor',
            '/api/statsmodel/plot/release',
            // ... other endpoints for openai/openai-cookbook
        ],

        // ... mappings for other repositories
    };
    const getCustomTextForEndpoint = (endpoint) => {
        // Example logic for custom text based on endpoint
        if (endpoint.includes('open')) {
            return 'maximum day issues created : ';
        } else if (endpoint.includes('closed')) {
            return 'maximum day issues closed: ';
        } else if (endpoint.includes('month')) {
            return 'maximum month  issues closed: ';
        }
    };

    const [imageUrls, setImageUrls] = useState([]);
    const [Text, setText] = useState([]);
    const fetchRepoData = async (repoName) => {
        setImageUrls([]);
        setText([])
        try {
            const endtext = textapi[repoName];
            const fetchPromisestext = endtext.map(endpoint =>
                fetch(`https://middleendflask-zjd2eijkya-uc.a.run.app${endpoint}?repo=${repoName}`)
                    .then(response => {
                        if (!response.ok) {
                            // Instead of throwing an error, return null or a similar marker
                            return null;
                        }
                        return response.text().then(text => getCustomTextForEndpoint(endpoint) + text);;
                    })
                    .catch(error => {
                        // Log the error and return a marker
                        console.error('Fetch error:', error);
                        return null;
                    })
            );

            const text = (await Promise.all(fetchPromisestext)).filter(url => url !== null);
            
            setText(text);
        } catch (error) {
            console.error('Failed to fetch text :', error);
        }

        try {
            const endpoints = repoEndpoints[repoName];
            const fetchPromises = endpoints.map(endpoint =>
                fetch(`https://middleendflask-zjd2eijkya-uc.a.run.app${endpoint}?repo=${repoName}`)
                    .then(response => {
                        if (!response.ok) {
                            // Instead of throwing an error, return null or a similar marker
                            return null;
                        }
                        return response.text();
                    })
                    .catch(error => {
                        // Log the error and return a marker
                        console.error('Fetch error:', error);
                        return null;
                    })
            );

            const urls = (await Promise.all(fetchPromises)).filter(url => url !== null);
            console.log("Fetched URLs:", urls);
            setImageUrls(urls);
        } catch (error) {
            console.error('Failed to fetch image URLs:', error);
        }

    };

    return (
        <>
            <nav>
                {REPOSITORIES.map(repo => (
                    <button key={repo} onClick={() => fetchRepoData(repo)}>
                        {repo}
                    </button>
                ))}
            </nav>
            {
                Text.map((value)=>(
                    <p>{value}</p>
                ))
                       
                
            }
            {imageUrls.map((url, index) => (
                <img key={index} src={url} alt={`Image for repository`} />
            ))}

        </>
    );
}

export default App;

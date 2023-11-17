import React from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const LineChartComponent = ({ data }) => {
    if (!data) {
        // Return null or some placeholder if data is not available
        return <p>Loading chart data...</p>;
    }
    // Process data for charting
    const labels = Object.keys(data);
    const values = Object.values(data);

    const chartData = {
        labels: labels,
        datasets: [
            {
                label: 'Issue Counts',
                data: values,
                fill: false,
                backgroundColor: 'rgb(75, 192, 192)',
                borderColor: 'rgba(75, 192, 192, 0.2)',
            },
        ],
    };

    const options = {
        scales: {
            y: {
                beginAtZero: true,
                ticks: {
                    // Additional configuration for y-axis ticks can be added here
                },
            },
            x: {
                // Configuration for x-axis can be added here if needed
            }
        },
        // Include other options here if needed
    };

    return <Line data={chartData} options={options} />;
};

export default LineChartComponent;

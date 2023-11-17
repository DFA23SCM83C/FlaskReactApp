import LineChartComponent from "../Components/LineChartComponent";
import React, {useState} from "react";



function IssuesPage(){
    const [data, setData] = useState(null);
    const jsonData = {
        "SeleniumHQ/selenium": 220,
        "angular/angular-cli": 165,
        "angular/material": 0,
        "d3/d3": 16,
        "facebook/react": 100,
        "golang/go": 932,
        "google/go-github": 22,
        "keras-team/keras": 127,
        "milvus-io/pymilvus": 41,
        "openai/openai-cookbook": 30,
        "openai/openai-python": 111,
        "openai/openai-quickstart-python": 0,
        "pallets/flask": 30,
        "sebholstein/angular-google-maps": 2,
        "tensorflow/tensorflow": 378
    }
    // useEffect(() => {
    //   // Replace '/api/issue' with the full URL path if needed
    //   fetch('http://localhost:3000/api/issues')
    //       .then((response) => response.json())
    //       .then((data) => setData(data))
    //       .catch((error) => console.error('Error fetching data:', error));
    // }, []);

    return (
        <>
            <h1>Issue Count Chart</h1>
            {jsonData ? <LineChartComponent data={jsonData} /> : <p>Loading data...</p>}
        </>
    )
}
export default IssuesPage;

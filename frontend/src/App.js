import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import IssuesPage from './Page/Issues';
import IssuesMonthPage from "./Page/IssuesMonth";
import IssuesWeekPage from "./Page/IssuesWeek";
import IssuesCreatedClosedPage from "./Page/IssuesCreatedClosed";
import ForkPage from "./Page/Fork";
import StarsPage from "./Page/Stars";
function App() {
    return (
        <BrowserRouter> {/* Wrap your routing logic with BrowserRouter */}
            <div className="App">
                <Routes> {/* Use Routes for defining Route paths */}
                    <Route path='/' element={<IssuesPage />} />
                    {/* <Route path='/week' element={<IssuesWeekPage />} />
                    <Route path='/month' element={<IssuesMonthPage />} />
                    <Route path='/created-closed' element={<IssuesCreatedClosedPage />} />
                    <Route path='/fork' element={<ForkPage />} />
                    <Route path='/star' element={<ForkPage />} /> */}
                </Routes>
            </div>
        </BrowserRouter>
    );
}

export default App;

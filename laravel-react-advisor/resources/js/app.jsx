import React, { useState } from 'react';

const App = () => {
    const [formData, setFormData] = useState({
        risk_tolerance: '',
        financial_goals: [],
        timeline: '',
        financial_standing: {
            income: 0,
            expenses: 0,
            savings: 0,
            debt: 0
        }
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({ ...formData, [name]: value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const response = await fetch('/api/client-data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData),
        });
        const result = await response.json();
        console.log(result);
    };

    return (
        <form onSubmit={handleSubmit}>
            <label>
                Risk Tolerance:
                <select name="risk_tolerance" onChange={handleChange}>
                    <option value="">Select</option>
                    <option value="Conservative">Conservative</option>
                    <option value="Moderate">Moderate</option>
                    <option value="Aggressive">Aggressive</option>
                </select>
            </label>
            <button type="submit">Submit</button>
        </form>
    );
};

export default App;

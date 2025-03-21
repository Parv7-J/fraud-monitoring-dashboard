import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const TimeSeriesGraph = () => {
  const [data, setData] = useState([]);
  const [timeFrame, setTimeFrame] = useState("1M"); // Default to 1 month

  useEffect(() => {
    fetchData(timeFrame);
  }, [timeFrame]);

  const fetchData = async (selectedTimeFrame) => {
    // Replace this with actual API call
    const response = await fetch(
      `/api/time-series?timeFrame=${selectedTimeFrame}`
    );
    const result = await response.json();
    setData(result);
  };

  return (
    <div>
      <h2>Fraud Trend Over Time</h2>
      <select value={timeFrame} onChange={(e) => setTimeFrame(e.target.value)}>
        <option value="1W">1 Week</option>
        <option value="1M">1 Month</option>
        <option value="3M">3 Months</option>
        <option value="1Y">1 Year</option>
      </select>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="predictedFrauds"
            stroke="#8884d8"
            name="Predicted Frauds"
          />
          <Line
            type="monotone"
            dataKey="reportedFrauds"
            stroke="#82ca9d"
            name="Reported Frauds"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default TimeSeriesGraph;

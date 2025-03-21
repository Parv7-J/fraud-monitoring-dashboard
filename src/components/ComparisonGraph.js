import React, { useState, useEffect } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ComparisonGraph = () => {
  const [data, setData] = useState({
    labels: [],
    datasets: [],
  });

  const [filters, setFilters] = useState({
    transactionChannel: "",
    paymentMode: "",
    gatewayBank: "",
    payerId: "",
    payeeId: "",
  });

  useEffect(() => {
    // Fetch data based on filters
    fetchData(filters);
  }, [filters]);

  const fetchData = async (filters) => {
    // Replace this with actual API call
    const response = await fetch("/api/fraud-comparison", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(filters),
    });
    const result = await response.json();

    setData({
      labels: result.labels,
      datasets: [
        {
          label: "Predicted Frauds",
          data: result.predictedFrauds,
          backgroundColor: "rgba(255, 99, 132, 0.5)",
        },
        {
          label: "Reported Frauds",
          data: result.reportedFrauds,
          backgroundColor: "rgba(53, 162, 235, 0.5)",
        },
      ],
    });
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: "Predicted vs Reported Frauds",
      },
    },
  };

  return (
    <div>
      <h2>Fraud Comparison</h2>
      <div>{/* Add filter inputs here */}</div>
      <Bar options={options} data={data} />
    </div>
  );
};

export default ComparisonGraph;

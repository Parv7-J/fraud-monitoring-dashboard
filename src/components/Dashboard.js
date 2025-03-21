import React, { useState, useEffect } from "react";
import { Grid, Paper, Typography } from "@mui/material";
import TransactionTable from "./TransactionTable";
import ComparisonGraph from "./ComparisonGraph";
import TimeSeriesGraph from "./TimeSeriesGraph";
import EvaluationSection from "./EvaluationSection";
import {
  subscribeToUpdates,
  unsubscribeFromUpdates,
} from "../services/websocket";

function Dashboard() {
  const [data, setData] = useState({
    transactions: [],
    comparisonData: {},
    timeSeriesData: [],
    evaluationMetrics: {},
  });

  useEffect(() => {
    subscribeToUpdates((newData) => {
      setData((prevData) => ({ ...prevData, ...newData }));
    });

    return () => {
      unsubscribeFromUpdates();
    };
  }, []);

  return (
    <Grid container spacing={3} sx={{ padding: 3 }}>
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>
          Transaction and Fraud Monitoring Dashboard
        </Typography>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <TransactionTable transactions={data.transactions} />
        </Paper>
      </Grid>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <ComparisonGraph data={data.comparisonData} />
        </Paper>
      </Grid>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <TimeSeriesGraph data={data.timeSeriesData} />
        </Paper>
      </Grid>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <EvaluationSection metrics={data.evaluationMetrics} />
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Dashboard;

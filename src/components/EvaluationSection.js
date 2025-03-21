import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Select,
  MenuItem,
  Typography,
} from "@mui/material";

const EvaluationSection = () => {
  const [metrics, setMetrics] = useState({
    confusionMatrix: { tp: 0, fp: 0, fn: 0, tn: 0 },
    precision: 0,
    recall: 0,
  });
  const [timePeriod, setTimePeriod] = useState("1M");

  useEffect(() => {
    fetchMetrics(timePeriod);
  }, [timePeriod]);

  const fetchMetrics = async (selectedTimePeriod) => {
    // Replace this with actual API call
    const response = await fetch(
      `/api/evaluation-metrics?timePeriod=${selectedTimePeriod}`
    );
    const result = await response.json();
    setMetrics(result);
  };

  const { confusionMatrix, precision, recall } = metrics;

  return (
    <div>
      <Typography variant="h5" gutterBottom>
        Model Evaluation
      </Typography>
      <Select
        value={timePeriod}
        onChange={(e) => setTimePeriod(e.target.value)}
        sx={{ mb: 2 }}
      >
        <MenuItem value="1W">1 Week</MenuItem>
        <MenuItem value="1M">1 Month</MenuItem>
        <MenuItem value="3M">3 Months</MenuItem>
        <MenuItem value="1Y">1 Year</MenuItem>
      </Select>
      <Typography variant="h6" gutterBottom>
        Confusion Matrix
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell></TableCell>
              <TableCell>Predicted Positive</TableCell>
              <TableCell>Predicted Negative</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            <TableRow>
              <TableCell>Actual Positive</TableCell>
              <TableCell>{confusionMatrix.tp}</TableCell>
              <TableCell>{confusionMatrix.fn}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Actual Negative</TableCell>
              <TableCell>{confusionMatrix.fp}</TableCell>
              <TableCell>{confusionMatrix.tn}</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>
      <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
        Metrics
      </Typography>
      <Typography>Precision: {precision.toFixed(2)}</Typography>
      <Typography>Recall: {recall.toFixed(2)}</Typography>
    </div>
  );
};

export default EvaluationSection;

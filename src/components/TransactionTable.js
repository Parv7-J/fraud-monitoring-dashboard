import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  TextField,
} from "@mui/material";
import { predictFraud } from "../services/api";

const TransactionTable = () => {
  const [transactions, setTransactions] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    // Fetch transactions (for simplicity, we're using predict endpoint for demo purposes)
    predictFraud({ transaction_id: "dummy", amount: 0 }).then((response) => {
      setTransactions([response.data]); // Replace with actual data fetching logic
    });
  }, []);

  const filteredTransactions = transactions.filter((txn) =>
    txn.transaction_id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div>
      <TextField
        label="Search by Transaction ID"
        variant="outlined"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        sx={{ mb: 2 }}
      />
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Transaction ID</TableCell>
              <TableCell>Amount</TableCell>
              <TableCell>Is Fraud</TableCell>
              <TableCell>Fraud Probability</TableCell>
              <TableCell>Fraud Source</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredTransactions.map((txn) => (
              <TableRow key={txn.transaction_id}>
                <TableCell>{txn.transaction_id}</TableCell>
                <TableCell>{txn.amount}</TableCell>
                <TableCell>{txn.is_fraud ? "Yes" : "No"}</TableCell>
                <TableCell>{txn.fraud_probability}</TableCell>
                <TableCell>{txn.fraud_source}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </div>
  );
};

export default TransactionTable;

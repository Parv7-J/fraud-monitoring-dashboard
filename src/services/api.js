import axios from "axios";

const API_BASE_URL = "https://084b-35-201-233-83.ngrok-free.app"; // Replace with your actual FastAPI URL

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "X-API-KEY":
      "42bcb9b5bfab7a69de851828c99908cf04a88f9ab449bb3b783ab4bda5650981", // Replace with your actual API key
  },
});

export const predictFraud = (transaction) =>
  apiClient.post("/predict", transaction);
export const batchPredictFraud = (batchRequest) =>
  apiClient.post("/batch-predict", batchRequest);
export const reportFraud = (report) => apiClient.post("/report-fraud", report);

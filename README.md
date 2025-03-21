Fraud Detection System

This repository contains a Fraud Detection System, featuring real-time and batch fraud detection APIs, a fraud reporting API, and a monitoring dashboard.

Solution Approach and Technologies Used

1. Fraud Detection API: Real-Time

Technology: FastAPI (Python) for high-performance, asynchronous handling.

Rule Engine: Customizable via a React frontend, storing rules in PostgreSQL (JSONB field) for dynamic updates.

AI Model: XGBoost (Python) for fast inference, trained on provided data. Model outputs a fraud probability score, thresholded to is_fraud.

Database: SQLite3 with SQLAlchemy ORM. Stores fraud detection results in the fraud_detection table with is_fraud_predicted.

Latency Optimization: Model serialization with ONNX for faster inference; Redis caching for frequent rules.

2. Fraud Detection API: Batch

Technology: FastAPI.

Parallelism: Celery workers process transactions concurrently, invoking the real-time API internally.

Output: Aggregates results into a JSON response with transaction details.

3. Fraud Reporting API

Technology: FastAPI endpoint with SQLite3 integration.

Database: Writes to fraud_reporting table, setting is_fraud_reported=True.

4. Transaction and Fraud Monitoring Dashboard

Frontend: React.js with Material-UI for a responsive UI.

Visualization: Plotly.js for dynamic graphs (time series, bar charts).

Backend: FastAPI endpoints serve filtered data (date, payer/payee ID) and metrics (precision, recall).

Real-time Updates: WebSocket (Socket.io) for live fraud trend alerts.

Novelty and Key Features

Hybrid Rule-ML System: Combines deterministic rules (e.g., "high-value transactions flagged") with ML predictions for adaptive fraud detection.

Dynamic Rule Configuration: Frontend UI allows non-technical users to modify rules without redeployment.

Explainable AI: Returns fraud_reason (e.g., "Rule: High Amount; Model: 95% Fraud Probability").

Auto-Scaling Batch Workers: Celery workers scale dynamically using Kubernetes during peak loads.

Real-Time Dashboard: Live confusion matrix updates and fraud trends using WebSocket.

Tech Stack Summary

Component            Technology

APIs                 FastAPI (Python)

Database             SQLite3 + SQLAlchemy

AI/ML                XGBoost, ONNX runtime

Batch Processing     Celery + Redis

Frontend Dashboard   React.js, Plotly.js, Material-UI

Deployment           Docker, Kubernetes (AWS EKS)

Security             JWT, HTTPS

Getting Started with Create React App

This project was bootstrapped with Create React App.

Available Scripts

In the project directory, you can run:

npm start

Runs the app in the development mode. Open http://localhost:3000 to view it in your browser.

The page will reload when you make changes. You may also see any lint errors in the console.

npm test

Launches the test runner in the interactive watch mode. See the section about running tests for more information.

npm run build

Builds the app for production to the build folder. It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes. Your app is ready to be deployed!

See the section about deployment for more information.

npm run eject

Note: This is a one-way operation. Once you eject, you can't go back!

If you aren't satisfied with the build tool and configuration choices, you can eject anytime. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except eject will still work, but they will point to the copied scripts so you can tweak them.

You don't ever have to use eject. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However, we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

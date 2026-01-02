# ticket-classification-sla-system
NLP-based customer support ticket classification with SLA tracking
# Customer Support Ticket Classification & SLA System

## Overview
This project is an end-to-end NLP-based customer support ticket classification system that automatically:
- Classifies support tickets
- Assigns business priority
- Computes SLA status (On Track / At Risk / Breached)
- Supports real-time and bulk CSV ticket processing

The system is built using Machine Learning and exposed via a FastAPI backend, simulating how real customer support platforms work.

---

## Business Problem
Customer support teams receive large volumes of tickets daily. Manual classification and prioritization lead to:
- Slow response times
- SLA breaches
- Poor customer satisfaction

This project automates ticket handling to improve efficiency and SLA compliance.

---

## Solution Approach

### Ticket Classification
- Technique: TF-IDF + Logistic Regression
- Categories: Billing inquiry, Technical issue, Refund request, etc.
- Focus on fast, interpretable baseline model suitable for production MVPs

### Priority Assignment
- High: Billing / Refund issues
- Medium: Technical issues
- Low: General inquiries

### SLA Logic
- High priority → 4 hours
- Medium priority → 8 hours
- Low priority → 24 hours

SLA status is calculated dynamically at prediction time.

---

## Features
- NLP-based ticket classification
- Priority and SLA calculation
- Real-time prediction API
- Bulk CSV upload endpoint
- FastAPI backend with Swagger UI

---

## Tech Stack
- Python
- Scikit-learn
- NLP (TF-IDF)
- FastAPI
- Pandas
- Uvicorn

---

## Project Structure
ticket_project/
├── app.py
├── training.ipynb
├── README.md

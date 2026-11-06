# Customer Churn Prediction – Data Engineering & Machine Learning

## Project Overview

This project aims to build an end-to-end data pipeline for customer churn analysis and prediction using CRM data extracted from Zoho CRM.

The objective is to transform raw CRM data into meaningful business insights through data engineering, machine learning, and data visualization.

## Data Engineering – Medallion Architecture

The project follows the Medallion Architecture using Databricks and Delta Lake:

- **Bronze Layer:** Ingestion and storage of raw CRM data (Accounts, Contacts, Deals).
- **Silver Layer:** Data cleaning, transformation, deduplication, and quality processing using SQL and PySpark.
- **Gold Layer:** Business-ready datasets organized using a Star Schema for analytical processing.

## Machine Learning – Churn Prediction

A **Random Forest Classifier** was developed using Scikit-learn to predict customer churn based on CRM data.

- Binary classification: Churn / No Churn.
- Feature preparation and model training.
- Model evaluation with a reported accuracy of **97.3%**.

## Data Visualization

Power BI dashboards were developed to analyze customer behavior, monitor churn indicators, and visualize prediction results using DAX measures.

## Technologies Used

- **Data Engineering:** Databricks, Apache Spark, PySpark, SQL
- **Storage:** Delta Lake
- **Data Modeling:** Medallion Architecture, Star Schema
- **Machine Learning:** Python, Scikit-learn, Random Forest
- **Visualization:** Power BI, DAX
- **Source Data:** Zoho CRM

## Project Workflow

```text
Zoho CRM
   |
   v
Bronze Layer (Raw Data)
   |
   v
Silver Layer (Cleaning & Transformation)
   |
   v
Gold Layer (Star Schema)
   |
   +-------------------+
   |                   |
   v                   v
Random Forest       Power BI
Churn Prediction    Dashboards

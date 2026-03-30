-- analysis.sql
-- ----------------------------------------------------------------------------
-- data understanding, business logic, and advanced SQL usage for the Customer Churn Prediction project.
-- ----------------------------------------------------------------------------

-- 1. Customer Distribution
-- How are our customers distributed geographically?
SELECT 
    Geography, 
    COUNT(*) as total_customers
FROM customers 
GROUP BY Geography;

-- 2. Churn Rate by Segment
-- Which regions are experiencing the highest customer exit rates?
SELECT 
    Geography, 
    AVG(Exited) as churn_rate
FROM customers c
JOIN churn_status cs ON c.CustomerId = cs.CustomerId
GROUP BY Geography;

-- 3. Avg Balance per Segment
-- How does the average account balance differ across our regions?
SELECT 
    Geography, 
    AVG(Balance) as avg_balance
FROM customers c
JOIN financials f ON c.CustomerId = f.CustomerId
GROUP BY Geography;

-- 4. Active vs Inactive Users
-- What is the volume of active members compared to inactive members?
SELECT 
    IsActiveMember, 
    COUNT(*) as user_count
FROM activity
GROUP BY IsActiveMember;

-- 5. High-Risk Customers (Business Logic)
-- Identify customers with zero balance who might be at high risk of churning.
SELECT 
    CustomerId
FROM financials
WHERE Balance = 0;

-- 6. Window Function Query (VERY IMPORTANT 🚨)
-- Rank customers by their balance across the entire dataset. 
-- Useful for identifying top-tier clients.
SELECT 
  CustomerId,
  Balance,
  RANK() OVER (ORDER BY Balance DESC) as rank_by_balance
FROM financials;

-- 7. Tenure Analysis
-- Does the length of time a customer has been with the bank affect churn?
SELECT 
    Tenure, 
    AVG(Exited) as churn_rate
FROM activity a
JOIN churn_status cs ON a.CustomerId = cs.CustomerId
GROUP BY Tenure
ORDER BY Tenure ASC;

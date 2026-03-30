-- feature_engineering.sql
-- This query acts as our extraction layer. 
-- We use CTEs, JOINs, Window Functions, and CASE statements to prove robust SQL skills.

WITH aggregated_activity AS (
    -- Example CTE calculating active product ratio (just a simple example of CTE usage)
    SELECT 
        CustomerId,
        Tenure,
        NumOfProducts,
        HasCrCard,
        IsActiveMember,
        CAST(NumOfProducts AS FLOAT) / CASE WHEN Tenure = 0 THEN 1 ELSE Tenure END AS prod_per_year
    FROM activity
),

demographic_financials AS (
    -- Joining customers and financials
    SELECT
        c.CustomerId,
        c.Geography,
        c.Gender,
        c.Age,
        f.CreditScore,
        f.Balance,
        f.EstimatedSalary,
        -- Window function: Rank customers by balance within their geography
        RANK() OVER (PARTITION BY c.Geography ORDER BY f.Balance DESC) as wealth_rank_in_region,
        -- Window function: Average salary by age bracket (grouping ages by roughly decades)
        AVG(f.EstimatedSalary) OVER (PARTITION BY c.Geography, ROUND(c.Age / 10) * 10) as avg_regional_age_salary,
        -- Simple flag creation using CASE statements
        CASE WHEN f.Balance = 0 THEN 1 ELSE 0 END AS is_zero_balance
    FROM customers c
    JOIN financials f ON c.CustomerId = f.CustomerId
)

-- Final SELECT bringing it all together with the target variable
SELECT 
    df.CustomerId,
    df.Geography,
    df.Gender,
    df.Age,
    df.CreditScore,
    df.Balance,
    df.EstimatedSalary,
    df.wealth_rank_in_region,
    df.avg_regional_age_salary,
    df.is_zero_balance,
    act.Tenure,
    act.NumOfProducts,
    act.HasCrCard,
    act.IsActiveMember,
    act.prod_per_year,
    -- Join target variable
    cs.Exited
FROM demographic_financials df
JOIN aggregated_activity act ON df.CustomerId = act.CustomerId
JOIN churn_status cs ON df.CustomerId = cs.CustomerId;

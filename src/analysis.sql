-- ============================================================
-- Customer Churn Analysis — SQL Queries
-- Author: Veda Sai Polisetty
-- Dataset: IBM Telco Customer Churn (7,032 customers)
-- ============================================================

-- 1. Overall churn rate
SELECT
    COUNT(*)                                          AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)  AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                 AS churn_rate_pct
FROM customers;

-- 2. Churn rate by contract type (KEY INSIGHT)
SELECT
    Contract,
    COUNT(*)                                                    AS customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)            AS churned,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                           AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;
-- Result: Month-to-month 42.7% | One year 11.3% | Two year 2.8%

-- 3. Churn rate by tenure bucket
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 12 THEN '0–12 months'
        WHEN tenure BETWEEN 13 AND 24 THEN '13–24 months'
        WHEN tenure BETWEEN 25 AND 48 THEN '25–48 months'
        ELSE '49–72 months'
    END                                                         AS tenure_group,
    COUNT(*)                                                    AS customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                           AS churn_rate_pct
FROM customers
GROUP BY tenure_group
ORDER BY churn_rate_pct DESC;

-- 4. Average monthly charges for churners vs stayers
SELECT
    Churn,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
    ROUND(AVG(tenure), 1)         AS avg_tenure_months,
    COUNT(*)                      AS customers
FROM customers
GROUP BY Churn;

-- 5. Churn rate by internet service type
SELECT
    InternetService,
    COUNT(*)                                                    AS customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                           AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;
-- Fiber optic customers churn at 2x the rate of DSL customers

-- 6. Impact of tech support on churn
SELECT
    TechSupport,
    COUNT(*)                                                    AS customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                           AS churn_rate_pct
FROM customers
WHERE InternetService != 'No'
GROUP BY TechSupport
ORDER BY churn_rate_pct DESC;

-- 7. High-risk segment: month-to-month + fiber + no tech support
SELECT
    COUNT(*)                                                    AS at_risk_customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                           AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)                              AS avg_monthly_charges
FROM customers
WHERE Contract       = 'Month-to-month'
  AND InternetService = 'Fiber optic'
  AND TechSupport    = 'No';
-- This segment churns at ~65% — prime retention target

-- 8. Revenue at risk from churners
SELECT
    SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END)  AS monthly_revenue_lost,
    SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END)
        * 12                                                       AS annual_revenue_at_risk
FROM customers;

-- 9. Churn by payment method
SELECT
    PaymentMethod,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1
    ) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- 10. Senior citizens churn rate vs non-seniors
SELECT
    CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-senior' END AS segment,
    COUNT(*)                                                          AS customers,
    ROUND(
        100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1
    )                                                                 AS churn_rate_pct
FROM customers
GROUP BY SeniorCitizen;

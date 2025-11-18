-- ============================================================
-- Credit Card Customers (BankChurners.csv) - Reload section
-- Context: SQL Server on Linux (Docker), VS Code MSSQL extension
-- CSV must exist inside the container at: /var/opt/mssql/data/BankChurners.csv
-- Prereq: Database [ChurnDB], schemas [stg],[ods], and tables [stg.BankChurners_raw], [ods.BankChurners]
-- ============================================================

/* Ensure the database exists (no-op if already created) */
IF DB_ID('ChurnDB') IS NULL
BEGIN
    CREATE DATABASE ChurnDB;
END
GO

/* Switch to project database */
USE ChurnDB;
GO

/* 1) Clear staging table and bulk-load the CSV (no CODEPAGE on Linux) */
TRUNCATE TABLE stg.BankChurners_raw;   -- If table does not exist, CREATE it first in your setup script

BULK INSERT stg.BankChurners_raw
FROM '/var/opt/mssql/data/BankChurners.csv'
WITH (
    FIRSTROW        = 2,        -- skip header
    FIELDTERMINATOR = ',',      -- CSV delimiter
    ROWTERMINATOR   = '0x0a',   -- Unix newline; if needed, change to '0x0d0a' (Windows)
    TABLOCK
);
GO

/* Quick sanity check */
SELECT TOP 5 * FROM stg.BankChurners_raw;
GO

/* 2) Refresh ODS table with clean, typed data */
TRUNCATE TABLE ods.BankChurners;  -- Keep the schema, just reload rows

INSERT INTO ods.BankChurners (
    CLIENTNUM, Attrition_Flag, Customer_Age, Gender, Dependent_count,
    Education_Level, Marital_Status, Income_Category, Card_Category,
    Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon,
    Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Avg_Open_To_Buy,
    Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1,
    Avg_Utilization_Ratio
)
SELECT
    TRY_CAST(NULLIF(LTRIM(RTRIM(CLIENTNUM)),'')                 AS INT),
    LTRIM(RTRIM(Attrition_Flag)),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Customer_Age)),'')              AS INT),
    LTRIM(RTRIM(Gender)),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Dependent_count)),'')           AS INT),
    LTRIM(RTRIM(Education_Level)),
    LTRIM(RTRIM(Marital_Status)),
    LTRIM(RTRIM(Income_Category)),
    LTRIM(RTRIM(Card_Category)),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Months_on_book)),'')            AS INT),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Total_Relationship_Count)),'')  AS INT),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Months_Inactive_12_mon)),'')    AS INT),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Contacts_Count_12_mon)),'')     AS INT),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Credit_Limit)),'')              AS FLOAT),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Total_Revolving_Bal)),'')       AS INT),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Avg_Open_To_Buy)),'')           AS FLOAT),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Total_Amt_Chng_Q4_Q1)),'')      AS FLOAT),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Total_Trans_Amt)),'')           AS INT),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Total_Trans_Ct)),'')            AS INT),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Total_Ct_Chng_Q4_Q1)),'')       AS FLOAT),
    TRY_CAST(NULLIF(LTRIM(RTRIM(Avg_Utilization_Ratio)),'')     AS FLOAT)
FROM stg.BankChurners_raw;
GO

/* 3) Basic validation */
SELECT COUNT(*) AS row_count FROM ods.BankChurners;
SELECT TOP 10 * FROM ods.BankChurners;

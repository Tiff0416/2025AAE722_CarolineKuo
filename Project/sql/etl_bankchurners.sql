/* ============================================================
   BankChurners.csv ETL script (Windows SQL Server version)
   1. Ensure schemas exist
   2. Rebuild staging table (text only)
   3. BULK INSERT from CSV
   4. Rebuild clean ODS table with proper types
   5. Load ODS from staging
   6. Basic validation
   ============================================================ */

USE ChurnDB;
GO

/* 1) Ensure schemas exist (run once is enough) */
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'stg')
    EXEC('CREATE SCHEMA stg');
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'ods')
    EXEC('CREATE SCHEMA ods');
GO

/* 2) Drop and recreate staging table with NVARCHAR columns */
IF OBJECT_ID('stg.BankChurners_raw', 'U') IS NOT NULL
    DROP TABLE stg.BankChurners_raw;
GO

CREATE TABLE stg.BankChurners_raw (
    CLIENTNUM                 NVARCHAR(255),
    Attrition_Flag            NVARCHAR(255),
    Customer_Age              NVARCHAR(255),
    Gender                    NVARCHAR(255),
    Dependent_count           NVARCHAR(255),
    Education_Level           NVARCHAR(255),
    Marital_Status            NVARCHAR(255),
    Income_Category           NVARCHAR(255),
    Card_Category             NVARCHAR(255),
    Months_on_book            NVARCHAR(255),
    Total_Relationship_Count  NVARCHAR(255),
    Months_Inactive_12_mon    NVARCHAR(255),
    Contacts_Count_12_mon     NVARCHAR(255),
    Credit_Limit              NVARCHAR(255),
    Total_Revolving_Bal       NVARCHAR(255),
    Avg_Open_To_Buy           NVARCHAR(255),
    Total_Amt_Chng_Q4_Q1      NVARCHAR(255),
    Total_Trans_Amt           NVARCHAR(255),
    Total_Trans_Ct            NVARCHAR(255),
    Total_Ct_Chng_Q4_Q1       NVARCHAR(255),
    Avg_Utilization_Ratio     NVARCHAR(255),
    NB1                       NVARCHAR(MAX),
    NB2                       NVARCHAR(MAX)
);
GO

/* 3) BULK INSERT CSV into staging table
      If your CSV path changes, only edit the FROM '...csv' line */
BULK INSERT stg.BankChurners_raw
FROM 'C:\Users\tingy\OneDrive\SQLData\BankChurners.csv'
WITH (
    FIRSTROW        = 2,         -- skip header
    FIELDTERMINATOR = ',',       -- CSV delimiter
    ROWTERMINATOR   = '0x0a',    -- Unix-style newline
    TABLOCK
);
GO

/* 4) Quick check that staging loaded correctly */
SELECT TOP 10 * FROM stg.BankChurners_raw;
SELECT COUNT(*) AS row_count_stg FROM stg.BankChurners_raw;
GO

/* 5) Drop and recreate clean ODS table */
IF OBJECT_ID('ods.BankChurners', 'U') IS NOT NULL
    DROP TABLE ods.BankChurners;
GO

CREATE TABLE ods.BankChurners (
    CLIENTNUM                 INT             NOT NULL,
    Attrition_Flag            NVARCHAR(50)    NOT NULL,
    Customer_Age              INT             NULL,
    Gender                    NVARCHAR(10)    NULL,
    Dependent_count           INT             NULL,
    Education_Level           NVARCHAR(50)    NULL,
    Marital_Status            NVARCHAR(50)    NULL,
    Income_Category           NVARCHAR(50)    NULL,
    Card_Category             NVARCHAR(50)    NULL,
    Months_on_book            INT             NULL,
    Total_Relationship_Count  INT             NULL,
    Months_Inactive_12_mon    INT             NULL,
    Contacts_Count_12_mon     INT             NULL,
    Credit_Limit              FLOAT           NULL,
    Total_Revolving_Bal       INT             NULL,
    Avg_Open_To_Buy           FLOAT           NULL,
    Total_Amt_Chng_Q4_Q1      FLOAT           NULL,
    Total_Trans_Amt           INT             NULL,
    Total_Trans_Ct            INT             NULL,
    Total_Ct_Chng_Q4_Q1       FLOAT           NULL,
    Avg_Utilization_Ratio     FLOAT           NULL
);
GO

/* 6) Load clean ODS table from staging
      Trim spaces, map empty strings to NULL, and cast types */
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

/* 7) Final validation */
SELECT COUNT(*) AS row_count_ods FROM ods.BankChurners;
SELECT TOP 10 * FROM ods.BankChurners;
GO

SELECT * FROM ods.BankChurners;


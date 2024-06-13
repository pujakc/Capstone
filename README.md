# Predicting Mortgage Rates During Economic Uncertainty


## Problem Statement
While purchasing real estate, most buyers opt for mortgages rather than paying full amount in cash. The interest rate on these mortgages is a critical factor, influencing buyer's ability to manage monthly payments through their cash flows from the property or other resources. Amidst rapidly rising mortgage rates, reaching historical highs not seen in decades, home buyers are facing significant uncertainty. To alleviate this confusion, this project aims to develop machine learning model to accurately  predict mortgage rates, enabling buyers to lock in most favorable rate. Besides individual home buyers , the resulting model will also serve broader range of stakeholders, including realstate firms, financial institutiona and policymakers, by providing insights into future rate trends.

The volatility in mortgage rates can create substantial challenges for prospective homebuyers, who must make crucial financial decision under uncertain conditions. An accurae prediction model can give buyers in timing their mortgage rate locks to secure better financial terms. Other statkeholders such as real estate firms can better manage their investmwnt strategies, financial institutions can adjust their lending practices, and policy makers  can understand market dynamics to craft informed policies.

Following concerns of various stakeholders like prospective buyers and real state agents will be addressed by this model:

* What is the best mortgage rate that a home buyer can lock without breaking their banks.
* Real Estate Agent: With this interest rate is it okay to invest on a property.


## Project Overview

This project involves developing and comparing multiple machine learning models, including Ordinary Linear Regression, Ensemble regression models and an LSTM(Long Short Term Memory)time series model to predict the 30 year fixed mortgage rate. Additional sentiment analysis is done on FOMC(Federal Open Market Committee)meeting minutes to visualize if these sentiments impact the increase or decrease of the interest rates.

 ### Data Collection: 
- Mortgage rates: Obtain historical mortgage rate data from sources like Freddie Mac.
- Housing market indicators: Data on home sales, home prices, housing inventory, and housing starts can be sourced from real estate databases, government housing agencies, or industry reports.
- Unemployment data: Obtain unemployment rates from government labor departments or economic databases.
- GDP growth: GDP growth data can be sourced from government economic agencies or financial databases.
- Federal Open market committee meeting minutes which was taken from  Kaggle

### Data Precprocessing: 
Clean and preprocess the collected data to handle missing values, outliers, and inconsistencies. Perform feature engineering to create meaningful features that capture temporal patterns and relationships

### Exploratory Data Analysis (EDA): 
Conduct EDA to understand the distribution of data, identify trends, and uncover correlations between variables and visualize trends over time.

### Feature Engineering: 
Create new features or transform existing ones to enhance the predictive power of the model.

### Sentiment Analysis

### Model Development : 
Experiment with various machine learning algorithms, including linear regression, random forests, gradient boosting machines, and neural networks, to build predictive models. Use cross-validation to ensure model stability and generalizability.

### Model Training: 
Split the data into training and testing sets. Train the model using historical data.

### Model Evaluation:
Evaluate the model's performance using metrics like root mean squared error (RMSE)  

### Hyperparameter Tuning: 
Fine-tune the model hyperparameters to improve performance.

### Feature Importance: 
Analyze feature importance to understand which variables have the most significant impact on mortgage rate predictions

### Deployment: 
Deploy the best-performing model to Github
### Documentation and Reporting: 

Document the entire process, including data sources, methodologies, model development, and evaluation. Prepare a comprehensive report and presentation to communicate the findings and insights.

By following these steps and considerations, we can develop an effective machine learning model to predict future mortgage rates based on historical trends and other economic indicators.
 

## Work Books
The workbooks should be read in the following order

- 01_data_collection_EDA.ipynb
In this notebook I have imported various historical data which was collected from publicly available data from the Federal Reserve Economic Date(FRED) published by Federal Reserve Bank of  St.Louis. The data collected from individual source was then combined into a single dataframe which was used for preliminary EDA and then exported as CSV file for future use for model creationa and evaluation


- 02_FOMC_SentimentAnalysis.ipynb
 
 In this notebook I have used the historical and current FOMC(Federal Open Market Committee) meeting minutes data set from Kaggle https://www.kaggle.com/datasets/vladtasca/fomc-meeting-statements-and-minutes to perform a Sentiment Analysis on the meeting minute and analyse the impact of positive or negative sentiment on interest rate

- 03_Model_creation.ipynb

In this notebook I have created a Linear Regression model and 3 Ensemble models RandomForestRegressor, AdaBoostRegressor and  GradientBoostingRegressorr for Mortgage rate prediction and chosed the best one for evaluation

- 04_LSTM_TimeseriesForecast_Model.ipynb
This notebook aims to develop LSTM model to predict future values for Mortgate Rate. Several models were created and evaluated to choose th best one



### Analysis

The analysis process involved importing data from aforementioned data sets. After importing an extensive exploratory data analysis was performed on the dataset to check for the any anomalies in data like checking for missing values, wrong data types assignments, unrealistic data which needs corerection before the data can be fed into any machine learning model. Extensive visualizations were done with various graphs and plots. Analysis and visualization techniques were used to find how are independent variables related to check if some features can be excluded. Also, checked if interaction between the features can be created. Python was used to import, cleaning ,analysing data. Trying different approaches to address the analysis question. Different visualization techniques were used to understand answer analysis questions. Analyzed the data trends. Identified features to be fed into machine learning algorithm. 

Correlation heatmap showed there was a very strong positive correlation between 30 year fixed MortgateRate(MORTGAGE30US) and various features like GS10 (10-Year Treasury Constant Maturity Rate) and FEDFUNDS(Federal Funds Rate). And strong negative correlation with GDP, CSUSHPISA(Case-Shiller U.S. National Home Price Index)
From the analysis and visualizationit was confirmed that the 30 year Mortgage Rate is not highest currently. The interest rates very high before 1988 and has decreased with ups and downs and it seems the mortgage rates were smalles after covid 19 in 2020 and then increasing gradually with slight up and down. We can also see the with least Mortgagerate there was highest unemployment so far.The housing affordability index was in peak during the after the 2008 recesson and was decreasing with ups and down since then. It was lowest right after the Covid 19

'DATE', 'MORTGAGE30US', 'FEDFUNDS', 'GS10', 'MICH', 'UNRATE', 'USREC',
       'CSUSHPISA', 'MSACSR', 'GDP'


## Data Sources:
The majority of data was collected by using publicly available data taken from the Federal Reserve Economic Data (FRED) published by the Federal Reserve Bank of St. Louis1. Below is the list of datasource

- The data was obtained from University of Michigan, University of Michigan: Inflation Expectation [MICH], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/MICH, May 28, 2024.
- Federal Reserve Bank of St. Louis, NBER based Recession Indicators for the United States from the Period following the Peak through the Trough [USREC], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/USREC, May 28, 2024.
- Board of Governors of the Federal Reserve System (US), Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis [GS10], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/GS10, May 28, 2024.
- U.S. Bureau of Labor Statistics, Unemployment Rate [UNRATE], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/UNRATE, May 26, 2024.
- Freddie Mac, 30-Year Fixed Rate Mortgage Average in the United States [MORTGAGE30US], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/MORTGAGE30US, May 26, 2024.
- Board of Governors of the Federal Reserve System (US), Federal Funds Effective Rate [FEDFUNDS], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/FEDFUNDS, May 28, 2024.
-  https://www.kaggle.com/datasets/vladtasca/fomc-meeting-statements-and-minutes



## Data Dictionary

|Feature|Type|Description|
|---|---|---|
|DATE|datetime|The Date when the economic indicator was collected|
|MORTGAGE30US|float|30 Year Average Fixed Mortgage Rate in the United States|
|FEDFUNDS|float|Federal Funds Effective Rate|
|GS10|float|Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis |
|MICH|float|University of Michigan: Inflation Expectation|
|UNRATE|float64|Unemployment Rate|
|USREC|float|NBER based Recession Indicators for the United States from the Period following the Peak through the Trough |
|CSUSHPISA|float|S&P CoreLogic Case-Shiller U.S. National Home Price Index|
|MSACSR|object|Monthly Supply of New Houses in the United States|
|GDP|float|Gross Domestic Product|

Sentiment Data

|Date|datetime|The date of the FOMC meeting or statement release in the format YYYYMMDD|
|Type|object|Indicator for the type of document. Statementor  Meeting minutes.|
|Text|object|The text content of each paragraph in the meeting minutes or statements.|


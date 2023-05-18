# ds_Salary_Prediction
# data science salary estimator

## Code and Resources Used 
**Python Version:** 3.10.6  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  

## attributes in salary datasets
*	Salary Estimate
*	Job Description
*	Rating
*	Company 
*	Location
*	Company Headquarters 
*	Company Size
*	Company Founded Date
*	Type of Ownership 
*	Industry
*	Sector
*	Revenue
*	Competitors 

## Data Cleaning
After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

*	Parsed numeric data out of salary 
*	Made columns for employer provided salary and hourly wages 
*	Removed rows without salary 
*	Parsed rating out of company text 
*	Made a new column for company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed founded date into age of company 
*	Made columns for if different skills were listed in the job description:
    * Python    
    * Excel  
    * AWS  
    * Spark 
*	Column for simplified job title and Seniority 
*	Column for description length 

## Data Analysis and Model Building 
I looked at the distributions of the data and the value counts for the various categorical variables then transformed the categorical variables into dummy variables.Split the data into train and tests sets with a test size of 20%.   

I tried five different model: Multiple Linear Regression, Lasso Regression, Random Forest Regression, Support Vector Regression and Gradient Boosting Regression   and evaluated them using Mean Absolute Error.
## Model performance
The Random Forest model outperformed the other approaches on the test and validation sets. 



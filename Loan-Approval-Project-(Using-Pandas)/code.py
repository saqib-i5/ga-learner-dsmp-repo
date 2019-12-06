# --------------
import pandas as pd
import numpy as np



bank = pd.read_csv(path)
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)

numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)


# --------------
# code starts here

banks = bank.drop(columns ='Loan_ID')
print(banks.isnull().sum())


bank_mode = banks.mode().iloc[0]
banks.fillna(bank_mode,inplace = True) 
#code ends here


# --------------
# Code starts here





avg_loan_amount = pd.pivot_table(banks,index = ['Gender', 'Married', 'Self_Employed'], values = 'LoanAmount',aggfunc='mean')



# code ends here



# --------------
# code starts here




loan_approved_se= banks[(banks['Self_Employed'] == 'Yes') & (banks['Loan_Status'] == 'Y')].count().iloc[0]

loan_approved_nse = banks[(banks['Self_Employed'] == 'No') & (banks['Loan_Status'] == 'Y')].count().iloc[0]


percentage_se = (loan_approved_se*100)/614
print(percentage_se)

percentage_nse = (loan_approved_nse*100)/614
print(percentage_nse)
# code ends here


# --------------
# code starts here

loan_term = banks['Loan_Amount_Term'].apply(lambda x: x/12)
loan_term

big_loan_term = loan_term.apply(lambda x: x>=25).value_counts().loc[True]



# code ends here


# --------------
# code starts here
loan_groupby =banks.groupby('Loan_Status')
loan_groupby = loan_groupby[['ApplicantIncome','Credit_History']]

mean_values = loan_groupby.mean()

# code ends here



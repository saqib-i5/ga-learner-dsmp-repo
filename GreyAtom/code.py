# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(path) 
loan_status = data['Loan_Status'].value_counts()

loan_status.plot(kind ='bar')


#Code starts here


# --------------
#Code starts here




property_and_loan = data.groupby(['Property_Area','Loan_Status']).size().unstack()
property_and_loan.plot(kind='bar', stacked=False, figsize=(15,10))

plt.xlabel("Property Area")
plt.ylabel('Loan Status')
plt.xticks(rotation=45, ha='right')
plt.show()




# --------------
#Code starts here

education_and_loan = data.groupby(['Education','Loan_Status']).size().unstack()
education_and_loan.plot(kind='bar', stacked=False, figsize=(15,10))

plt.xlabel("Education Status")
plt.ylabel('Loan Status')
plt.xticks(rotation=45, ha='right')
plt.show()


# --------------
#Code starts here
graduate = data[data['Education'] == 'Graduate']
not_graduate= data[data['Education'] == 'Not Graduate']

graduate.plot(kind='density', label='Graduate')
not_graduate.plot(kind='density', label='Not Graduate')











#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
fig ,(ax_1,ax_2,ax_3)= plt.subplots(nrows = 3 , ncols = 1)

ax_1 = data.plot.scatter(x='ApplicantIncome',y ='LoanAmount')
ax_1.set_title('Applicant Income')

ax_2 = data.plot.scatter(x='CoapplicantIncome',y ='LoanAmount')
ax_2.set_title('CoapplicantIncome')


data['TotalIncome']= data['ApplicantIncome'] + data['CoapplicantIncome']

ax_3 = data.plot.scatter(x='TotalIncome',y ='LoanAmount')
ax_3.set_title('TotalIncome')





import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

#Loading data
df = pd.read_csv(r'C:\Users\Massimo Camuso\Desktop\Academics\Spring 2025\CIS 3715 (Principles of Data Science)\Final Project\Loan_default.csv')
df.info()
df.isnull()

#Visualize numerical features
#df['Age'].hist(bins=60)
#df['Income'].hist(bins='auto')
#df['LoanAmount'].hist(bins='auto')
#df['CreditScore'].hist(bins='auto')
#df['MonthsEmployed'].hist(bins='auto')
#df['NumCreditLines'].hist(bins='auto')
#df['InterestRate'].hist(bins='auto')
#df['LoanTerm'].hist(bins='auto')
#df['DTIRatio'].hist(bins='auto')
#df['Default'].hist(bins='auto')
#plt.show()

'''
#-------------Scale/encode necessary numerical features--------
#Credit Score min/max

CS_min = df['CreditScore'].min(axis=0)
CS_max = df['CreditScore'].max(axis=0)

df['CreditScore'] = (df['CreditScore']-CS_min)/(CS_max-CS_min)
#print(df['CreditScore'])


#DTI Ratio min/max 
DTI_min = df['DTIRatio'].min(axis=0)
DTI_max = df['DTIRatio'].max(axis=0)

df['DTIRatio'] = (df['DTIRatio']-DTI_min)/(DTI_max-DTI_min)
#print(df['DTIRatio']) 




#OneHotEncoding for LoanTerm
onehotencoder = OneHotEncoder(sparse_output=False)

loanterm_encoded = onehotencoder.fit_transform(df[['LoanTerm']])
loanterm_df = pd.DataFrame(
    loanterm_encoded,
    columns=onehotencoder.get_feature_names_out(['LoanTerm'])
)

df = pd.concat([df.drop('LoanTerm', axis=1), loanterm_df], axis=1)
print(loanterm_df.head())



#-----------Convert categorical features to numerical features---------
#-----Education------
#print(df['Education'].unique())

default_rates = df.groupby('Education')['Default'].mean().sort_values()

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=default_rates.index, y=default_rates.values, order=default_rates.index)
plt.title('Default Rate by Education Level')
plt.ylabel('Default Rate (%)')
plt.xlabel('Education')
plt.show()
#^^ Clear trend; higher level of education, the less the default rate. Label encoding


education_order = [['PhD', "Master's", "Bachelor's", 'High School']]  # Note: Nested list for sklearn

encoder = OrdinalEncoder(categories=education_order)
df['Education'] = encoder.fit_transform(df[['Education']])


# Verify
print(encoder.categories_)  # Should show: [['PhD', 'Masters', 'Bachelors', 'High School']]




#----EmploymentType------
#print(df['EmploymentType'].unique())
# Calculate default rates by EmploymentType
employment_default_rates = df.groupby('EmploymentType')['Default'].mean().sort_values()

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(
    x=employment_default_rates.index,
    y=employment_default_rates.values,
    order=employment_default_rates.index  # Orders bars by default rate
)
plt.title('Default Rate by Employment Type')
plt.ylabel('Default Rate (%)')
plt.xlabel('Employment Type')

# Rotate x-labels if needed
plt.xticks(rotation=45)  # Helps with longer category names
plt.show()

# Define the order explicitly (important!)
employment_order = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']

# Create a mapping dictionary
employment_mapping = {k: v for v, k in enumerate(employment_order)}

# Apply mapping
df['EmploymentType'] = df['EmploymentType'].map(employment_mapping)

# Verify
#print(df[['EmploymentType', 'EmploymentType_Encoded']].head(15))
'''
#-----Martial Status-------
print(df['MaritalStatus'].unique())

# Calculate default rates
marital_defaults = df.groupby('MaritalStatus')['Default'].mean().sort_values()

# Plot
plt.figure(figsize=(8, 4))
sns.barplot(
    x=marital_defaults.index,
    y=marital_defaults.values,
    order=marital_defaults.index  # Sort by default rate
)
plt.title('Default Rate by Marital Status')
plt.ylabel('Default Rate (%)')
plt.xticks(rotation=45)
plt.show()
'''
# Define your custom risk order (low to high)
marital_order = ['Married', 'Single', 'Divorced']

# Create mapping dictionary
marital_mapping = {'Married': 0, 'Single': 1, 'Divorced': 2}

# Apply to column
df['MaritalStatus'] = df['MaritalStatus'].map(marital_mapping)

# Verify
#print(df[['MaritalStatus', 'MaritalStatus_Encoded']].head())

#---HasMortgage----
#print(df['HasMortgage'].unique())
#print(df.groupby('HasMortgage')['Default'].mean())

df['HasMortgage'] = df['HasMortgage'].map({'Yes': 0, 'No': 1})
#print(df['HasMortgage'].head(10))



#-----HasDependents------
#print(df['HasDependents'].unique())
#print(df.groupby('HasDependents')['Default'].mean())
df['HasDependents'] = df['HasDependents'].map({'Yes': 0, 'No': 1})
#print(df['HasDependents'].head(10))




#-----LoanPurpose-----
print(df['LoanPurpose'].unique())


# Calculate default rates by LoanPurpose
purpose_defaults = df.groupby('LoanPurpose')['Default'].mean().sort_values()

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(
    x=purpose_defaults.index,
    y=purpose_defaults.values,
    order=purpose_defaults.index  # Sort by default rate
)
plt.title('Default Rate by Loan Purpose')
plt.ylabel('Default Rate (%)')
plt.xlabel('Loan Purpose')
plt.xticks(rotation=45)  # Rotate labels if needed
#plt.show()




#Onehot since there is no clear trend in defaults
loanpurpose_encoded = onehotencoder.fit_transform(df[['LoanPurpose']])
loanpurpose_df = pd.DataFrame(
    loanpurpose_encoded,
    columns=onehotencoder.get_feature_names_out(['LoanPurpose'])
)

df = pd.concat([df.drop('LoanPurpose', axis=1), loanpurpose_df], axis=1)
#print(loanpurpose_df.head())





#-----HasCoSigner------
print(df['HasCoSigner'].unique())
print(df.groupby('HasCoSigner')['Default'].mean())
df['HasCoSigner'] = df['HasCoSigner'].map({'Yes': 0, 'No': 1})
#print(df['HasCoSigner'].head(10))





#----------------------------Check Correlation btwn numericals------------------------------
df_numeric = df.select_dtypes(include=['number'])
corr = df_numeric.corr()

# Plot as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numeric Features')
#plt.show()



'''
#---------------------Export preprocessed data---------------------------------------------
#csv_path = r'C:\Users\Massimo Camuso\Desktop\preprocessed_loan_data.csv'
#df.to_csv(csv_path, index=False)



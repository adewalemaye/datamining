# %% [markdown]
# ## Adewale Maye
# ## HW EDA

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm6103 as dm 

# %%
# import dataframe
dfhappy = dm.api_rfit('Happy') 

# %%
# Question 1
print(dfhappy.columns)
dfhappy.rename(columns= {'year':'Year', 'id':'ID', 'hrs1':'Hrs', 'marital' : 'Marital', 'childs' : 'Num_children', 'income' : 'Family_income_tot', 'happy' : 'Happiness', 'ballet' : 'Ballot'}, inplace = True)
print(dfhappy.columns)

# %%
# Question 2
dfhappy.info()
print(dfhappy.value_counts())
print(dfhappy['Hrs'].unique)
print(dfhappy['Family_income_tot'].unique)
print(dfhappy['Num_children'].unique)


# There are 2 integer columns and 6 object columns. Looking at the columns of Hrs, Family_income_tot, and Num_children, the family
# income column and hours columns have string values.

# %%
# Question 3

dfhappy.loc[dfhappy["Hrs"].isin(["Not applicable"]), "Hrs"] = np.NaN
dfhappy.replace({"Not applicable": np.NaN, "No answer": np.NaN, "Don't know": np.NaN}, inplace=True)
dfhappy["Hrs"] = pd.to_numeric(dfhappy["Hrs"])


# %%
# Question 4

value_counts = dfhappy["Num_children"].value_counts()
print(value_counts)

## There are two types of strings in this column: Dk na and Eight or m. Dk na is a string value response that should be treated as nonresponse.


def cleanDfchildren(row):
  children = row["Num_children"]
  try: children = int(children) 
  except: pass
  
  try: 
    if not isinstance(children,int) : children = float(children)  # no change if already int, or if error when trying
  except: pass
  
  if ( isinstance(children,int) or isinstance(children,float) ) and not isinstance(children, bool): return ( children if children>=0 else np.nan )
  if isinstance(children, bool): return np.nan
  # else: # assume it's string from here onwards
  children = children.strip()
  if children == "Dk na": return np.nan
  if children == "Eight or m": 
    # strategy
    # let us just randomly distribute it, say according to chi-square, 
    # deg of freedom = 2 (see distribution from https://en.wikipedia.org/wiki/Chi-square_distribution for example) 
    # values peak at zero, most range from 0-5 or 6. Let's cap it 100
    children = min(89 + 2*np.random.chisquare(2) , 100)
    return children # leave it as decimal
  return np.nan # catch all, just in case
# end function cleanGssAge
print("\nReady to continue.")

# %%
# Question 5

def cleanDfFamilyIncome(row, Family_income_tot):
    thisamt = row[Family_income_tot]
    if isinstance(thisamt, str):
        thisamt = thisamt.strip()
        if (thisamt == "Not applicable"): return np.nan
        if (thisamt == "Refused"): return np.nan 
        if (thisamt == "Lt $1000"): return np.random.uniform(0,999)
        if (thisamt == "$1000 to 2999"): return np.random.uniform(1000,2999)
        if (thisamt == "$3000 to 3999"): return np.random.uniform(3000,3999)
        if (thisamt == "$4000 to 4999"): return np.random.uniform(4000,4999)
        if (thisamt == "$5000 to 5999"): return np.random.uniform(5000,5999)
        if (thisamt == "$6000 to 6999"): return np.random.uniform(6000,6999)
        if (thisamt == "$7000 to 7999"): return np.random.uniform(7000,7999)
        if (thisamt == "$8000 to 9999"): return np.random.uniform(8000,9999)
        if (thisamt == "$10000 - 14999"): return np.random.uniform(10000,14999)
        if (thisamt == "$15000 - 19999"): return np.random.uniform(15000,19999)
        if (thisamt == "$20000 - 24999"): return np.random.uniform(20000,24999)
        if (thisamt == "$25000 or more"): return ( 25000 + 10000*np.random.chisquare(2) )
    return np.nan
print("\nReady to continue.")

# %%
# Question 6

print(dfhappy['Happiness'].describe())


## recode the values in the 'Happiness' column
recoding_dict = {'Very happy': 2, 'Pretty happy': 1, 'Not too happy': 0}
dfhappy['Happiness'] = dfhappy['Happiness'].map(recoding_dict).fillna(np.nan)

# %%
# Question 7

print(dfhappy['Ballot'].unique())
dfhappy['Ballot'] = dfhappy['Ballot'].str.replace('Ballot ', '').map({'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd'})
print(dfhappy['Ballot'].unique())


# %%
# Question 8

import matplotlib.pyplot as plt
import seaborn as sns


dfhappy_clean = dfhappy.dropna(subset=['Marital', 'Hrs'])
dfhappy_clean.boxplot(column='Hrs', by='Marital')
plt.xlabel('Marital Status')
plt.ylabel('Hours Worked per Week')
plt.show()

sns.violinplot(x="Happiness", y="Family_income_tot", data=dfhappy)

# %%
# Question 9

%pip install scipy
import scipy.stats as stats

## Filter out null values
df = dfhappy[['Marital', 'Hrs']].dropna()

## Create a dictionary of dataframes for each marital status
groups = {key: group['Hrs'] for key, group in df.groupby('Marital')}

## Perform the ANOVA test
f_value, p_value = stats.f_oneway(groups['Never married'], groups['Married'], groups['Widowed'], groups['Divorced'])

## Print the results
print('F-value:', f_value)
print('p-value:', p_value)


## Using the ANOVA test, the null hypothesis indicates that the means of the hours worked per week are equal for all marital statuses. /n 
## If the p-value is less than the significance level (e.g., 0.05), we reject the null hypothesis and conclude that at least one group /n
## differs significantly from the others in terms of hours worked per week. Given the p-value is less than 0.05, we reject the null hypthesis /n
## and conclude that at least one group differs significantly from the others in terms of hours worked per week.

# %%
# Question 10

from scipy.stats import chi2_contingency
# Create a contingency table of Num_children and Happiness
contingency_table = pd.crosstab(dfhappy['Num_children'], dfhappy['Happiness'])

# Conduct chi-square test of independence
chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

# Print the results
print('Chi-square statistic:', chi2_stat)
print('p-value:', p_val)
print('Degrees of freedom:', dof)
print('Expected frequencies:', expected)



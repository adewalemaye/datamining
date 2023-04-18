# %% [markdown]
# # HW Pandas - grades
# ## By: Adewale Maye
# ### Date: February 27, 2022

# %%
pip install rfit

# %%
import rfit
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## Question 1

# %%
dats = rfit.dfapi('Dats_grades')
rfit.dfchk(dats)
print(dats.info())

# The variables in the df are homework, quiz, and project grades for a DATS class and the data types for these variables are float64 data types.

dats.mean()

# %% [markdown]
# ## Question 2

# %%
HWAvg = dats.iloc[:,0:8].mean(axis=1) * 10
dats.insert(loc = 8,
            column = 'HWavg',
            value = HWAvg)

dats.head() 

# %% [markdown]
# ## Question 3

# %%
weights = {'H1': 0.03, 'H2': 0.03, 'H3': 0.03, 'H4': 0.03, 'H5': 0.03, 'H6': 0.03, 'H7': 0.03, 'H8': 0.03,
           'Q1': 0.1, 'Q2': 0.15, 'Proj1': 0.2, 'Proj2': 0.25}

total_grade = (dats['H1'] * weights['H1'] +
               dats['H2'] * weights['H2'] +
               dats['H3'] * weights['H3'] +
               dats['H4'] * weights['H4'] +
               dats['H5'] * weights['H5'] +
               dats['H6'] * weights['H6'] +
               dats['H7'] * weights['H7'] +
               dats['H8'] * weights['H8'] +
               dats['Q1'] * weights['Q1'] +
               dats['Q2'] * weights['Q2'] +
               dats['Proj1'] * weights['Proj1'] +
               dats['Proj2'] * weights['Proj2']) * 100

dats['total'] = total_grade

dats['total'] = total_grade


# %%
dats.insert(loc = 14,
            column = 'total',
            value = total_grade)

dats.head() 

# %% [markdown]
# ## Question 4

# %%
dats.mean()

# %% [markdown]
# ## Question 5

# %%
import os
dats.to_csv('grades.csv', index=True)


# %% [markdown]
# ## Question 6

# %%
grade = " "

for course in courses:
    if course["Grade"] >= 93:
        grade = "A"
    elif course["Grade"] >= 91 and course["Grade"] < 93:
        grade = "A-"
    elif course["Grade"] >= 84.5 and course["Grade"] < 86:
        grade = "B"
    elif course["Grade"] >= 77.5 and course["Grade"] < 80:
        grade = "C+"
    else:
        grade = "D+"

return grade

# %% [markdown]
# ## Question 7

# %%
def find_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

dats['grade'] = dats['total'].apply(find_grade)



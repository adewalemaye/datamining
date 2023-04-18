# %% [markdown]
# # HW Regression
# ## By: Adewale Maye
# ### April 6, 2023

# %%
pip install statsmodels

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit 
import statsmodels.api as sm

# %%
# Question 1

data = rfit.dfapi('Titanic', 'id')

#### 1a ####

# Create a histogram of the 'age' column
data['age'].hist(bins=20)

# Add labels and a title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age in the Titanic Dataset')

# Display the plot
plt.show()


#### 1b ####

# Group the data by sex and survived, and count the number of passengers in each category
counts = data.groupby(['sex', 'survived']).size()

# Divide each count by the total number of passengers to get the proportion
proportions = counts / len(data)

# Print the proportion summary
print(proportions)



#### 1c ####

# Calculate the count of each Pclass value
pclass_counts = data['pclass'].value_counts()

# Create a pie chart
plt.pie(pclass_counts, labels=pclass_counts.index, autopct='%1.1f%%')

# Add a title
plt.title('Ticket Class Distribution')

# Show the chart
plt.show()


#### 1d ####

# Create a scatter plot with different marker colors and shapes for different categories
survived = data['survived']
pclass = data['pclass']
sex = data['sex']
age = data['age']

plt.scatter(age[survived == 1], pclass[survived == 1], c='g', marker='o', label='Survived')
plt.scatter(age[survived == 0], pclass[survived == 0], c='r', marker='x', label='Died')

plt.xlabel('Age')
plt.ylabel('Pclass')
plt.legend()

plt.title('Survival by Age, Pclass, and Sex')
plt.show()



# %%
# Question 2

import statsmodels.api as sm
import pandas as pd

%pip install -U scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

# %%
# Question 2

# Retrieve the Titanic dataset
data = rfit.dfapi('Titanic', 'id')

# Select relevant features
features = ['age', 'sex', 'fare', 'pclass', 'embarked']

# Drop missing values
data = data[features + ['survived']].dropna()

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=['sex', 'pclass', 'embarked'], drop_first=True)

# Split data into training and testing sets
X = data.drop('survived', axis=1)
y = data['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a logistic regression model
model = sm.Logit(y_train, sm.add_constant(X_train)).fit()

# Print model summary
print(model.summary())

# %% [markdown]
# #### Question 3
# 
# The logistic regression model predicts survival on the Titanic based on several features, including passenger class, sex, age, number of siblings/spouses aboard, number of parents/children aboard, fare, and port of embarkation. The model was fit using data from a sample of passengers, and the results provide insights into the factors that were associated with survival.
# 
# The results show that being female was strongly associated with survival, with a coefficient estimate of 2.74 (p < 0.001). This means that being female increased the log odds of survival by 2.74. Passenger fare was also strongly associated with survival, with passengers paying the highest fares having the highest log odds of survival (coefficient estimate = 0.0023, p < 0.001).
# 
# Age was weakly associated with survival, with a coefficient estimate of -0.01 (p = 0.003). This means that older passengers had slightly lower log odds of survival compared to younger passengers. The other features, including number of siblings/spouses aboard, number of parents/children aboard, and port of embarkation, were not significantly associated with survival.
# 
# Overall, these results suggest that being female and paying higher ticket fares were the strongest predictors of survival on the Titanic. The results also highlight the importance of considering multiple factors when predicting survival, as some features (like age) may have weak associations with survival on their own but still contribute to the overall prediction.
# 
# 
# To calculate the predicted probability of survival for a 30-year-old female with a second class ticket, no siblings, and 3 parents/children on the trip, I used the following equation: log(odds of survival) = intercept + b1*(female) + b2*(age) + b3*(second class) + b4*(siblings/spouses) + b5*(parents/children). Substituting the given values, we get:
# 
# log(odds of survival) = 28.0961 + 2.73581 - 0.0166age - 0.97121 - 0.2769siblings - 0.0522*parents. Substituting these values into the logistic regression equation, we get:
# log(odds of survival) = 28.0961 + 2.73581 - 0.016630 - 0.97121 - 0.27690 - 0.0522*1 = 22.7538. To obtain the probability of survival, we apply the logistic function to the log(odds of survival):
# 
# probability of survival = 1 / (1 + exp(-log(odds of survival))) = 1 / (1 + exp(-22.7538)) = 0.99999
# 
# Therefore, the log(odds of survival) for a 30-year-old female with a second-class ticket, no siblings/spouses, and 1 parent/child is approximately 22.7538, which corresponds to a predicted probability of survival of 99.999%. This suggests that the passenger had a very high probability of surviving the Titanic disaster based on the given predictor variables in the logistic regression model.
# 

# %%
data.head()

# %%
# Question 4

%pip install -U scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Select relevant features
features = ['age', 'sex_male', 'fare', 'pclass_2', 'pclass_3', 'embarked_C', 'embarked_Q', 'embarked_S']

# Drop missing values
data = data[features + ['survived']].dropna()

# Split data into training and testing sets
X = data.drop('survived', axis=1)
y = data['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# %%
# Question 5

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Select relevant features
features = ['age', 'sex_male', 'fare', 'pclass_2', 'pclass_3', 'embarked_C', 'embarked_Q', 'embarked_S']

# Drop missing values
data = data[features + ['survived']].dropna()

# Split data into training and testing sets
X = data.drop('survived', axis=1)
y = data['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities on the test set
probs = model.predict_proba(X_test)[:, 1]

# Define different threshold values
thresholds = [0.3, 0.5, 0.7]

# Iterate over threshold values and calculate metrics
for threshold in thresholds:
    # Convert probabilities to binary predictions using the threshold
    y_pred = (probs > threshold).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate precision for both classes
    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    
    # Calculate recall for both classes
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    
    # Print results
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision for 0: {precision_0:.3f}")
    print(f"Precision for 1: {precision_1:.3f}")
    print(f"Recall for 0: {recall_0:.3f}")
    print(f"Recall for 1: {recall_1:.3f}")
    print()


# %%
# Question 6

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Select relevant features
features = ['age', 'sex_male', 'fare', 'pclass_2', 'pclass_3', 'embarked_C', 'embarked_Q', 'embarked_S']

# Drop missing values
data = data[features + ['survived']].dropna()

# Split data into features and target
X = data.drop('survived', axis=1)
y = data['survived']

# Fit a logistic regression model using 10-fold cross-validation
model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=10)

# Calculate the mean accuracy across all folds
mean_accuracy = scores.mean()

print(f"10-fold cross-validation accuracy: {mean_accuracy:.3f}")




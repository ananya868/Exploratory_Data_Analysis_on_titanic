import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import accuracy_score


df = pd.read_csv('train.csv')
print(df.head())

# Missing data
# print(df.isnull())
# cheking missing data using seaborn
# sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.set_style('whitegrid')
# sns.countplot(x='Survived', hue='Pclass', data=df, palette='RdBu_r')

# sns.distplot(df['Age'].dropna(), kde=False, color='darkred', bins=40)
# sns.countplot(x='SibSp', data=df)
# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Pclass', y='Age', data=df, palette='winter')
# plt.show()


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# applying this function
df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)

# checking the heatmap again
# sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# plt.show()

df.drop('Cabin', axis=1, inplace=True)
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

embark = pd.get_dummies(df['Embarked'], drop_first=True)
sex = pd.get_dummies(df['Sex'], drop_first=True)
df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
print(df.head())

df = pd.concat([df, sex, embark], axis=1)
print(df.head())


# DATA CLEANING is DONE

"""Building the Logistic Regression model"""
# Train Test split
df.drop('Survived', axis=1)

x_train, y_train, x_test, y_test = train_test_split(df.drop('Survived', axis=1),
                                                    df['Survived'], test_size=0.30, random_state=101)

# Training and Predicting
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

predictions = logmodel.predict(x_test)
acc = accuracy_score(y_test, predictions)
print(acc)







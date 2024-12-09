from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns

csvyears = ["2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]  
data_frames = [pd.read_csv(f"NBAMVP{i}.csv") for i in csvyears]
mvp_data = pd.concat(data_frames)

## Gets rid of T's
mvp_data['Rank'] = (
    mvp_data['Rank']
    .astype(str)
    .str.extract('(\d+)') 
    .astype('Int64')
)

## Features I would like to create visualizations for
old_features = ['G', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%', '3P%', 'FT%', 'MP', 'Share']

## Creates table of average statistics for players who won MVP
mvp_data_first = mvp_data[mvp_data['Rank'] == 1]
print("MVP Average Stats:")
print(mvp_data_first[old_features].mean())

## Barplot of stats with correlation to shares
mvp_corr = mvp_data[old_features]
correlation_with_share = mvp_corr.corr()['Share'].drop('Share')
plt.figure()
sns.barplot(x=np.abs(correlation_with_share).index, y=np.abs(correlation_with_share).values, color="darkorange")
sns.set_theme(style="whitegrid")
plt.title("Correlation between Statistics and MVP Votes")
plt.xlabel('Season Statistics')
plt.ylabel('Share of MVP Votes')
plt.show()

## Features that I will continue to use
features = ['G', 'PTS', 'TRB', 'AST', 'STL', 'BLK', 'FG%']

## Linear model
X = mvp_data[features] 
y = mvp_data['Share']

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=.9, random_state=42)
lr = linear_model.LinearRegression() 
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

mse = metrics.mean_squared_error(y_test, y_pred)
print(mse)
print(np.sqrt(mse))

## reading current season stats
currentMVP = pd.read_csv("currentMVP.csv")
## Saving for Later
names = currentMVP['Player']
currentMVP = currentMVP[features]
predictions = lr.predict(currentMVP)

## softmax scale the predictions so they add up to 1
predictions_scaled = predictions * 5
softmax_shares = np.exp(predictions_scaled) / np.sum(np.exp(predictions_scaled))

currentMVP['Predicted Share'] = softmax_shares
currentMVP['Predicted Share'] = (currentMVP['Predicted Share']*100).round(2)
currentMVP['Player'] = names
predicted_MVP_shares = currentMVP[['Player', 'Predicted Share']]
plt.table(cellText=predicted_MVP_shares.values, colLabels=predicted_MVP_shares.columns, loc='center', colColours=['lightgray', 'lightgray'])
plt.axis("off")
plt.title("Predicted Shares of MVP Votes as a Percentage")
plt.show()
plt.axis("on")

plt.pie(currentMVP['Predicted Share'])
plt.legend(currentMVP['Player'], title="Players", loc="center left", bbox_to_anchor=(1, .5))
plt.title("Predicted MVP Share Distribution")
plt.show()

## SECOND MODEL, follows the same formula

X = mvp_data[features]
mvp_data['Rank'] = (mvp_data['Rank'] == 1).astype(int)
y = mvp_data['Rank']

## Logistic Model
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=.9, random_state=42)
lr = linear_model.LogisticRegression() 
lr.fit(x_train, y_train)

# Model Effectiveness
y_pred = lr.predict_proba(x_test)[:, 1]
y_pred_classes = lr.predict(x_test)
predictions = [1 if i > 0.5 else 0 for i in y_pred]
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_pred_classes, predictions))
log_likelihood = np.sum(y_test * np.log(y_pred) + (1 - y_test) * np.log(1 - y_pred))
print("Log-Likelihood:")
print(log_likelihood)

currentMVP = pd.read_csv("currentMVP.csv")
names = currentMVP['Player']
currentMVP = currentMVP[features]

currentMVP['MVP Probability'] = lr.predict_proba(currentMVP[features])[:, 1]

## Make it sum to 1
prob_sum = currentMVP['MVP Probability'].sum()
currentMVP['MVP Probability'] = currentMVP['MVP Probability'] / prob_sum 

currentMVP['MVP Probability'] = (currentMVP['MVP Probability']*100).round(2)
currentMVP['Player'] = names

predicted_MVP_prob = currentMVP[['Player', 'MVP Probability']]
plt.table(cellText=predicted_MVP_prob.values, colLabels=predicted_MVP_prob.columns, loc='center', colColours=['lightgray', 'lightgray'])
plt.axis("off")
plt.title("Predicted Shares of MVP Votes as a Percentage")
plt.show()
plt.axis("on")

plt.pie(currentMVP['MVP Probability'])
plt.legend(currentMVP['Player'], title="Players", loc="center left", bbox_to_anchor=(1, .5))
plt.title("MVP Probability")
plt.show()
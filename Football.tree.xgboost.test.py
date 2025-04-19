import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import seaborn as sns
import os
from scipy.stats import uniform, randint

file_path = "C:\\Users\\theoa\\Downloads\\new data.csv"
df = pd.read_csv(file_path)
df = df.drop(columns=["Date"], errors="ignore")

df['MatchNumber'] = range(1, len(df) + 1)

# we now Keep betting odds for final evaluation
odds_columns = ["B365H", "B365D", "B365A"]

columns_to_drop = [
    "BFH", "BFD", "BFA", "BSH", "BSD", "BSA", "BWH", "BWD", "BWA"
]
df = df.drop(columns=columns_to_drop, errors='ignore')


elo_ratings = {
    "Man City": 2404,
    "Arsenal": 2259,
    "Liverpool": 2233,
    "Chelsea": 2123,
    "Newcastle": 2104,
    "Man United": 2102,
    "Tottenham": 2077,
    "Aston Villa": 2044,
    "Crystal Palace": 2036,
    "Brighton": 2009,
    "West Ham": 2002,
    "Fulham": 1992,
    "Everton": 1986,
    "Brentford": 1973,
    "Bournemouth": 1962,
    "Wolves": 1946,
    "Leicester": 1911,
    "Nott'm Forest": 1903,
    "Ipswich": 1858,
    "Southampton": 1858,
}
elo_dict = {team: elo_ratings.get(team, 1500) for team in set(df['HomeTeam']).union(set(df['AwayTeam']))}

def update_elo(home, away, home_goals, away_goals):
    K = 40
    home_elo = elo_dict[home]
    away_elo = elo_dict[away]

    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    expected_away = 1 - expected_home

    if home_goals > away_goals:
        outcome_home, outcome_away = 1, 0
    elif home_goals < away_goals:
        outcome_home, outcome_away = 0, 1
    else:
        outcome_home, outcome_away = 0.5, 0.5

    elo_dict[home] += K * (outcome_home - expected_home)
    elo_dict[away] += K * (outcome_away - expected_away)


for index, row in df.iterrows():
    update_elo(row['HomeTeam'], row['AwayTeam'], row['FTHG'], row['FTAG'])
    df.at[index, 'HomeElo'] = elo_dict[row['HomeTeam']]
    df.at[index, 'AwayElo'] = elo_dict[row['AwayTeam']]

def get_numeric_form(team, match_num, df):
    recent_matches = df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & (df['MatchNumber'] < match_num)].tail(3)
    return sum(1 if row['FTR'] == 'H' else -1 if row['FTR'] == 'A' else 0 for _, row in recent_matches.iterrows())


df['HomeForm'] = df.apply(lambda row: get_numeric_form(row['HomeTeam'], row['MatchNumber'], df), axis=1)
df['AwayForm'] = df.apply(lambda row: get_numeric_form(row['AwayTeam'], row['MatchNumber'], df), axis=1)

def rolling_avg(team, column, match_num, df):
    recent_matches = df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & (df['MatchNumber'] < match_num)].tail(3)
    return np.mean([row[column] if row['HomeTeam'] == team else row[column.replace('H', 'A')] for _, row in
                    recent_matches.iterrows()])



df['HomeAvgShots'] = df.apply(lambda row: rolling_avg(row['HomeTeam'], 'HS', row['MatchNumber'], df), axis=1)
df['AwayAvgShots'] = df.apply(lambda row: rolling_avg(row['AwayTeam'], 'HS', row['MatchNumber'], df), axis=1)
df['HomeAvgFouls'] = df.apply(lambda row: rolling_avg(row['HomeTeam'], 'HF', row['MatchNumber'], df), axis=1)
df['AwayAvgFouls'] = df.apply(lambda row: rolling_avg(row['AwayTeam'], 'HF', row['MatchNumber'], df), axis=1)


df['FTR'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})

features = ['HomeElo', 'AwayElo', 'HomeForm', 'AwayForm', 'HomeAvgShots', 'AwayAvgShots', 'HomeAvgFouls', 'AwayAvgFouls']

train_df = df.iloc[0:290]
predict_df = df.iloc[290:300]  # Now predicting known games to bet on

X_train = train_df[features]
y_train = train_df['FTR']
X_predict = predict_df[features]

# using optimal XGBoost parameters
best_params = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.5
}

xgb_best = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_best.fit(X_train, y_train)

# Predict match outcomes (290–300)
predicted = xgb_best.predict(X_predict)
actual = predict_df['FTR'].values
predict_df['Predicted'] = predicted

# Simulate betting
profit = 0
for idx, row in predict_df.iterrows():
    pred = row['Predicted']
    actual = row['FTR']
    if pred == 2:  # Bet on home win
        odds = row.get("B365H", 0)
        profit += (odds - 1) if actual == 2 else -1
    elif pred == 1:  # Bet on draw
        odds = row.get("B365D", 0)
        profit += (odds - 1) if actual == 1 else -1
    elif pred == 0:  # Bet on away win
        odds = row.get("B365A", 0)
        profit += (odds - 1) if actual == 0 else -1

result_map = {2: 'Home Win', 1: 'Draw', 0: 'Away Win'}


predict_df['PredictedLabel'] = predict_df['Predicted'].map(result_map)
predict_df['ActualLabel'] = predict_df['FTR'].map(result_map)
predict_df['PredictedOdds'] = predict_df.apply(
    lambda row: row['B365H'] if row['Predicted'] == 2 else row['B365D'] if row['Predicted'] == 1 else row['B365A'], axis=1
)
predict_df['Profit'] = predict_df.apply(
    lambda row: row['PredictedOdds'] - 1 if row['Predicted'] == row['FTR'] else -1, axis=1
)

display_columns = [
    'HomeTeam', 'AwayTeam', 'PredictedResult', 'ActualResult',
    'B365H', 'B365D', 'B365A', 'PredictedOdds', 'Profit'
]

predict_df['PredictedOdds'] = predict_df['PredictedOdds'].round(2)
predict_df['Profit'] = predict_df['Profit'].round(2)
predict_df[['B365H', 'B365D', 'B365A']] = predict_df[['B365H', 'B365D', 'B365A']].round(2)


print("Prediction vs Actual Outcomes & Betting Profit:")
print(predict_df[display_columns].to_string(index=False))

total_profit = predict_df['Profit'].sum()
print(total_profit)



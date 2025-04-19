import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import plot_tree
import os

file_path = "C:\\Users\\theoa\\Downloads\\E0.csv"
df = pd.read_csv(file_path)
df = df.drop(columns=["Date"], errors="ignore")

df['MatchNumber'] = range(1, len(df) + 1)
columns_to_drop = [
    "B365H", "B365D", "B365A", "BFH", "BFD", "BFA", "BSH", "BSD", "BSA", "BWH", "BWD", "BWA"
]
df = df.drop(columns=columns_to_drop, errors='ignore')

# Initial Elo ratings
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

df_filtered = df[df['MatchNumber'] > 10]
features = ['HomeElo', 'AwayElo', 'HomeForm', 'AwayForm', 'HomeAvgShots', 'AwayAvgShots', 'HomeAvgFouls',
            'AwayAvgFouls']
df_filtered['FTR'] = df_filtered['FTR'].map({'H': 2, 'D': 1, 'A': 0})

X = df_filtered[features]
y = df_filtered['FTR']
# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

os.makedirs("rfplots", exist_ok=True)

importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
# Feature Importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(features)[indices], palette='viridis')
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("rfplots/feature_importance.pdf")
plt.close()

#Partial Dependence Plots
selected_features = ['HomeElo', 'AwayElo', 'HomeForm', 'AwayForm']
for target_class in [0, 1, 2]:
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(
        rf_classifier,
        X_test,
        features=selected_features,
        target=target_class,
        ax=ax
    )
    plt.tight_layout()
    plt.savefig(f"rfplots/partial_dependence_class_{target_class}.pdf")
    plt.close()
#A single tree from the random forest
plt.figure(figsize=(15, 15))
plot_tree(rf_classifier.estimators_[0],
          filled=True,
          max_depth=2,
          feature_names=features,
          class_names=['A', 'D', 'H'],
          fontsize=14,
          proportion=True,
          rounded=True)
plt.title("Random Forest - Tree 1 (Depth = 2)")
plt.savefig("rfplots/tree_visual2.pdf", bbox_inches="tight")
plt.close()


cm = confusion_matrix(y_test, y_pred)

#Confusion Matrix plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Away Win', 'Draw', 'Home Win'], yticklabels=['Away Win', 'Draw', 'Home Win'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("rfplots/confusion_matrix.pdf")
plt.close()

#classification report
report = classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win'])
with open("rfplots/classification_report.txt", "w") as f:
    f.write(report)




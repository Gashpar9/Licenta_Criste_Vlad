import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier




# We have to merge the dataset containing attack data and the dataset containing data under normal operation

path_1 = os.path.join(os.getcwd(), "../../datasets/WADI_attackdataLABLE.csv")
path_2 = os.path.join(os.getcwd(), "../../datasets/WADI_14days_new.csv")

df_1 = pd.read_csv(path_1, header = 1, sep = ',')
df_2 = pd.read_csv(path_2, header = 0, sep = ',')

# Row column is trivial, Date and Time columns are not relevant for testing so we drop them

df_1 = df_1.drop(columns = ['Row ', 'Date ', 'Time'])
df_2 = df_2.drop(columns = ['Row', 'Date', 'Time'])

# Initially the normal operation dataset did not have an attack label feature so we have to add it ourselves

df_2['Attack LABLE (1:No Attack, -1:Attack)'] = 1

df = pd.concat([df_1, df_2], axis = 0)

# We also have to drop the following columns because they only have missing values

df = df.drop(columns = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS'])

# We use the fillna pandas method with a forward filling strategy for the rest of the missing values -> strategy = LAST VALUE ABOVE

df.ffill(inplace=True)

# Prepare and split data

X = df.drop('Attack LABLE (1:No Attack, -1:Attack)', axis=1)
y = df['Attack LABLE (1:No Attack, -1:Attack)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




# Training the models

model_abclf = AdaBoostClassifier(algorithm='SAMME', random_state=42)
model_abclf.fit(X_train, y_train)

model_bclf = BaggingClassifier(random_state=42)
model_bclf.fit(X_train, y_train)

model_dtclf = DecisionTreeClassifier(random_state=42)
model_dtclf.fit(X_train, y_train)

model_rfclf = RandomForestClassifier(random_state=42)
model_rfclf.fit(X_train, y_train)

model_etclf = ExtraTreesClassifier(random_state=42)
model_etclf.fit(X_train, y_train)

model_gbclf = GradientBoostingClassifier(random_state=42)
model_gbclf.fit(X_train, y_train)

model_hgbclf = HistGradientBoostingClassifier(random_state=42)
model_hgbclf.fit(X_train, y_train)




# Save trained models

joblib.dump(model_abclf, 'model_abclf.pkl')
joblib.dump(model_bclf, 'model_bclf.pkl')
joblib.dump(model_dtclf, 'model_dtclf.pkl')
joblib.dump(model_rfclf, 'model_rfclf.pkl')
joblib.dump(model_etclf, 'model_etclf.pkl')
joblib.dump(model_gbclf, 'model_gbclf.pkl')
joblib.dump(model_hgbclf, 'model_hgbclf.pkl')

# Save the test split data so the time isn't influenced by overfitting

test_data = {'X_test': X_test, 'y_test': y_test}
joblib.dump(test_data, 'test_data.pkl')
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


if __name__ == '__main__':
    ### Load data
    df = pd.read_csv('internship_train.csv')
    test = pd.read_csv('internship_hidden_test.csv')
    X = df.drop('target', axis=1)
    y = df['target'].copy()

    ### Scaling data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    test_scaled = scaler.transform(test)

    ### Fitting model
    rf = RandomForestRegressor()
    rf.fit(X_scaled, y)

    ### Prediction
    pred = rf.predict(test_scaled)
    answ = pd.DataFrame(pred)
    answ.to_csv('prediction.csv')
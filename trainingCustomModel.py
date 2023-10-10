import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle



df = pd.read_csv('coords.csv')
# which is the class . This is a multiclass classification where we want to establish relation betwee n our class and the features
X = df.drop('class', axis=1) # features

y = df['class'] # target value


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# After this line we will start training our model

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}


fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model



#exporting and serializing

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))

with open('sign_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)
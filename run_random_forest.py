import pandas
import kfold_template

import matplotlib.pyplot as pyplot

from sklearn.ensemble import RandomForestClassifier


dataset = pandas.read_csv("temperature_data.csv")

# print(dataset)
dataset = pandas.get_dummies(dataset)

# print(dataset)
print(dataset.columns)
# print(dataset.describe())

dataset=dataset.sample(frac=1).reset_index(drop=True)

print(dataset)

target = dataset['actual'].values
data_raw = dataset.drop('actual', axis = 1)
data = data_raw.values

feature_list = data_raw.columns


print(target)
print(data)

machine = RandomForestClassifier(criterion="gini", max_depth = 10, n_estimators = 300 ,bootstrap=True, max_features = "auto")
results = kfold_template.run_kfold(data, target, 4, machine, 0, 0)
print(results)



machine = RandomForestClassifier(criterion="gini", max_depth = 10, n_estimators = 300 ,bootstrap=True, max_features = "auto")
machine.fit(data, target)
feature_importances_raw = list(machine.feature_importances_)
print(feature_importances_raw)
print(feature_list)
feature_zip = zip(feature_list, feature_importances_raw)
# print(feature_zip)
feature_importances = [(feature, round(importance,4)) for feature, importance in zip(feature_list, feature_importances_raw)]
feature_importances = sorted(feature_importances, key = lambda x:x[1])
[print('{:13} : {}'.format(*feature_importance)) for feature_importance in feature_importances]


x_values = list(range(len(feature_importances_raw)))
y_values = feature_importances_raw
pyplot.bar(x_values, y_values)
pyplot.xticks(x_values, feature_list, rotation='vertical')
pyplot.ylabel("Feature Importance")
pyplot.xlabel('Feature')
pyplot.title("Feature Importance")
pyplot.tight_layout()
pyplot.savefig("feature_importances.png")
pyplot.close()






	

















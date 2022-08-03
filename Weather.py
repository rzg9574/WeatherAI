from ctypes import py_object
from modulefinder import IMPORT_NAME
import pandas as pd
import matplotlib.pyplot as pyplot
from matplotlib import style
from pyparsing import Combine
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

weather = pd.read_csv("WeatherDataFrom1975.csv", index_col="DATE")
def create_model():
    # print(weather.apply(pd.isnull).sum() / weather.shape[0])

    important_weather = weather[[
        "PRCP", 
        "SNOW", 
        "SNWD", 
        "TMAX", 
        "TMIN"]].copy()

    for i in important_weather.columns:
        important_weather[i].fillna(method='ffill', inplace=True)

    # print(important_weather.apply(pd.isnull).sum() / important_weather.shape[0])

    important_weather.index =  pd.to_datetime(important_weather.index)

    important_weather.apply(lambda x: (x ==9999).sum())


    #just seeing what the data looks like 
    # style.use("ggplot")
    # pyplot.scatter(important_weather.index, important_weather["PRCP"])
    # pyplot.xlabel("date")
    # pyplot.ylabel("Rain")
    # pyplot.show()

    important_weather.groupby(important_weather.index.year).sum()["PRCP"]

    important_weather["TargetTMAX"] = important_weather.shift(-1)["TMAX"]

    important_weather = important_weather.iloc[:-1, :].copy()

    important_weather["month_max"] = important_weather["TMAX"].rolling(30).mean()
    important_weather["month_day_max"] = important_weather["month_max"] / important_weather["TMAX"] 
    important_weather["max_min"] = important_weather["TMAX"] / important_weather["TMIN"] 
    important_weather["month_avg"] = important_weather["TMAX"].groupby(important_weather.index.month).apply(lambda x: x.expanding(1).mean())
    important_weather["avg_day_of_year_temp"] = important_weather["TMAX"].groupby(important_weather.index.day_of_year).apply(lambda x: x.expanding(1).mean())
    important_weather = important_weather.iloc[30:,:].copy()
    
    return important_weather



# train = important_weather.loc[:"2020-12-31"]
# test = important_weather.loc["2021-01-01":]

# reg.fit(train[predictors], train["TargetTMAX"])

# Ridge(alpha=0.1)

# predictions = reg.predict(test[predictors])

# # print("max temp", mean_absolute_error(test["TargetTMAX"], predictions))
# combined = pd.concat([test["TargetTMAX"], pd.Series(predictions, index=test.index)], axis=1)
# # pyplot.plot(combined)
# # pyplot.show()

# print(reg.coef_)


def create_predictions(predictors, important_weather,):
    reg = Ridge(alpha=0.1)
    train = important_weather.loc["1995-1-1":"2021-12-31"]
    test = important_weather.loc["2022-07-27":"2022-07-27"]
    reg.fit(train[predictors], train["TargetTMAX"])
    predictions = reg.predict(test)
    error = mean_absolute_error(test["TargetTMAX"], predictions)
    combined = pd.concat([test["TargetTMAX"], pd.Series(predictions, index=test.index)], axis=1)
    
    return error, combined

model = create_model() 

predictors = ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "month_max", "month_day_max", "max_min", "month_avg", "avg_day_of_year_temp"]

error, combined = create_predictions(predictors, model)

print(combined)

# print(important_weather.corr()["TargetTMAX"])
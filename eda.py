from sqlalchemy import create_engine, func
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

import pandas as pd

import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

from config import connection

# goals:
# explore distributions on air metrics, life-expectancy, & ecosystem
# explore correlation between air quality on humans and environment & life-expectancy, eco-system

## EXTRACT 

# Connect to database and create an epi table object
engine = create_engine(connection)
Base = automap_base()

Base.prepare(engine, reflect=True)

epi_country = Base.classes.epi_country
session = Session(engine)

# pull rows from your database
rows = session.query(
    epi_country.country, 
    epi_country.geo_subregion, 
    epi_country.ecosystem, 
    epi_country.air_h, 
    epi_country.air_e).all()

df_epi = pd.DataFrame(rows, columns=["country", "subregion", "ecosystem", "air_h", "air_e"])
df_hdi = pd.read_csv("data/HDI.csv")

## TRANSFORM

# check for null values in life-expect and drop
print(df_hdi["Life expectancy"].isna().sum())
df_hdi.dropna(subset=["Life expectancy"], inplace=True)
print(df_hdi["Life expectancy"].isna().sum())

# lowercase all countries in df_hdi
df_hdi.columns = [col.lower() for col in df_hdi.columns]

# save this table to postgres
df_hdi.to_sql("hdi", engine, if_exists='replace')

# filter to only western hemisphere
# limits data, removing
'''
epi_sa = df_epi[(df_epi["subregion"] == "South America") | 
            (df_epi["subregion"] == "North America") | 
            (df_epi["subregion"] == "Caribbean") | 
            (df_epi["subregion"] == "Meso America")]
'''

# join w/human development
joined_df = df_epi.merge(df_hdi, on='country', how='left')

# joining introduces NaN values, so we must clean again
print(joined_df["life expectancy"].isna().sum())
joined_df.dropna(subset=["life expectancy"], inplace=True)
print(joined_df["life expectancy"].isna().sum())

## EDA on air-health vs life-expectancy

# generate k-s test on air health distribution
ks_air_e = stats.kstest(joined_df["air_h"], 'norm')
ks_air_h = stats.kstest(joined_df["air_e"], 'norm')

print("K-S Test for Normality on Air_H", ks_air_h)
print("K-S Test for Normality on Air_E", ks_air_e)

# analyze visually & save figures
# histogram (bins varying)
joined_df["air_h"].plot.hist(bins=30)
plt.savefig("images/Air_H_dist30.png")
plt.clf()

joined_df["air_e"].plot.hist(bins=30)
plt.savefig("images/Air_E_dist30.png")
plt.clf()

joined_df["air_h"].plot.hist(bins=20)
plt.savefig("images/Air_H_dist20.png")
plt.clf()

joined_df["air_e"].plot.hist(bins=20)
plt.savefig("images/Air_E_dist20.png")
plt.clf()

# qq-plot
stats.probplot(joined_df["air_h"], dist="norm", plot=plt)
plt.savefig("images/Air_H_qq.png")
plt.clf()

stats.probplot(joined_df["air_e"], dist="norm", plot=plt)
plt.savefig("images/Air_E_qq.png")
plt.clf()

# generate k-s test on life-expectancy
ks_life = stats.kstest(joined_df["life expectancy"], 'norm')

print("K-S Test for Normality on life expectancy", ks_life)

joined_df["life expectancy"].plot.hist(bins=20)
plt.savefig("images/life_dist20.png")
plt.clf()

joined_df["life expectancy"].plot.hist(bins=30)
plt.savefig("images/life_dist30.png")
plt.clf()

stats.probplot(joined_df["life expectancy"], dist="norm", plot=plt)
plt.savefig("images/life_qq.png")
plt.clf()

# analyze relationships between life & air
scatter = sns.scatterplot(data=joined_df, x="air_h", y="life expectancy")
fig = scatter.get_figure()
fig.savefig("images/airh_hvs_life.png") 
plt.clf()

scatter = sns.scatterplot(data=joined_df, x="air_e", y="life expectancy")
fig = scatter.get_figure()
fig.savefig("images/aire_hvs_life.png") 
plt.clf()

# analyze relationships between air & ecosystem
scatter = sns.scatterplot(data=joined_df, x="air_e", y="ecosystem")
fig = scatter.get_figure()
fig.savefig("images/air_evs_eco.png") 
plt.clf()

# analyze relationships between air & air
scatter = sns.scatterplot(data=joined_df, x="air_e", y="air_h")
fig = scatter.get_figure()
fig.savefig("images/airevsairh.png") 
plt.clf()

# load your data back in
joined_df.to_sql("epi_hdi", engine, if_exists='replace')
engine.dispose()

# observations: seems to be positive correlation b/w air & human life expectancy

# split into test & train set
# prep training set by reshaping
X = joined_df["air_h"].array.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, joined_df["life expectancy"], test_size=0.25, random_state=42)

# validate with linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

r_sq = model.score(X_test, y_test)
print(f"R-squared score: {r_sq}")

# this statistic describes the proportion of variability described by the independent variable
# generally we are looking for a high r-score (not too high!) to find the goodness of fit
# since our r-score is under 50%, we might want to introduce more predictors or use a different model

# further analyze goodness of fit
y_pred = model.predict(X_test)

print("MAE", metrics.mean_absolute_error(y_test, y_pred))

# let's try this again with scaled data
scaler = MinMaxScaler()

joined_focus = joined_df[['air_h', 'life expectancy']]

scaled_data = scaler.fit_transform(joined_focus)
epihdi_scaled = pd.DataFrame(scaled_data, columns=joined_focus.columns)

X = epihdi_scaled["air_h"].array.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, epihdi_scaled["life expectancy"], test_size=0.25, random_state=42)

model.fit(X_train, y_train)

r_sq = model.score(X_test, y_test)
print(f"R-squared score: {r_sq}")

# further analyze goodness of fit
y_pred = model.predict(X_test)

print("MAE", metrics.mean_absolute_error(y_test, y_pred))

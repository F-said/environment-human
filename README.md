# environment-human

A Python pipeline to analyze predictors of life-expectancy using features from the `epi` dataset. Data is loaded in via `sql-alchemy` and transformed using `pandas`. Visualizations are generated via `seaborn` and `matplotlib` before data is loaded back into the `postgres` local databse.

## Results

![air human vs air env](airevsairh.png)
Correlation between air affects on humans & air affects on environment is not visually obvious.

![air human vs life](air_evs_eco.png)
Similarly, there does not appear to be a trend associated with air_e and life expectancy

![air human vs life](airh_hvs_life.png)
However, there is a positive correlation between air affect on human and life expectancy.

## Next Actions

* Find granular data that describes disease from air quality
* Modularize code
* Explore different models for more predictive power
* Load data into Tableau visaulization
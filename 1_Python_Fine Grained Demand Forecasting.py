# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/fine-grained-demand-forecasting. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/demand-forecasting.

# COMMAND ----------

# MAGIC %md The objective of this notebook is to illustrate how we might generate a large number of fine-grained forecasts at the store-item level in an efficient manner leveraging the distributed computational power of Databricks.  This is a Spark 3.x update to a previously published notebook which had been developed for Spark 2.x.  **UPDATE** marks in this notebook indicate changes in the code intended to reflect new functionality in either Spark 3.x or the Databricks platform.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC For this exercise, we will make use of an increasingly popular library for demand forecasting, [prophet](https://facebook.github.io/prophet/), which we will load into the notebook session using the %pip magic command. 

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install prophet

# COMMAND ----------

# MAGIC %md ## Step 1: Examine the Data
# MAGIC
# MAGIC For our training dataset, we will make use of 5-years of store-item unit sales data for 50 items across 10 different stores.  This data set is publicly available as part of a past Kaggle competition and can be downloaded with the `./config/Data Extract` notebook with your own Kaggle credentials.
# MAGIC
# MAGIC Once downloaded, we can upload the decompressed *train.csv* content to the data lake bronze layer. With the dataset accessible within Databricks, we can now explore it in preparation for modeling:

# COMMAND ----------

# MAGIC %run "./config/Data Extract"

# COMMAND ----------

# DBTITLE 1,Imports and Constants
import datetime
import logging
import math

import pandas as pd
import prophet
from pyspark.sql import functions, types
import shutil
from sklearn import metrics

# constant values
catalog = "fgdf_accelerator_dev"
bronze_file_path = f"/Volumes/{catalog}/bronze/kaggle/train.csv"
temporary_training_view = "tmp_kaggle_train"
silver_table = f"{catalog}.silver.store_item_history"

# COMMAND ----------

# DBTITLE 1,Ingest Data from the Bronze File (CSV) into a Temp View
# structure of the training data set
train_schema = types.StructType([
  types.StructField("date", types.DateType()),
  types.StructField("store", types.IntegerType()),
  types.StructField("item", types.IntegerType()),
  types.StructField("sales", types.IntegerType())
])

# read the training file into a dataframe
train_df = spark.read.csv(
  bronze_file_path, 
  schema=train_schema,
  header=True
)

# show data
display(train_df)

# make the dataframe queryable as a temporary view
train_df.createOrReplaceTempView(temporary_training_view)

# COMMAND ----------

# MAGIC %md When performing demand forecasting, we are often interested in general trends and seasonality.  Let's start our exploration by examining the annual trend in unit sales:

# COMMAND ----------

# DBTITLE 1,View Yearly Trends
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   YEAR(date) AS year, 
# MAGIC   SUM(sales) AS sales
# MAGIC FROM tmp_kaggle_train
# MAGIC GROUP BY YEAR(date)
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md It's very clear from the data that there is a generally upward trend in total unit sales across the stores. If we had better knowledge of the markets served by these stores, we might wish to identify whether there is a maximum growth capacity we'd expect to approach over the life of our forecast.  But without that knowledge and by just quickly eyeballing this dataset, it feels safe to assume that if our goal is to make a forecast a few days, months or even a year out, we might expect continued linear growth over that time span.
# MAGIC
# MAGIC Now let's examine seasonality.  If we aggregate the data around the individual months in each year, a distinct yearly seasonal pattern is observed which seems to grow in scale with overall growth in sales:

# COMMAND ----------

# DBTITLE 1,View Monthly Trends
# MAGIC %sql
# MAGIC
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') AS month,
# MAGIC   SUM(sales) AS sales
# MAGIC FROM tmp_kaggle_train
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# MAGIC %md Aggregating the data at a weekday level, a pronounced weekly seasonal pattern is observed with a peak on Sunday (weekday 0), a hard drop on Monday (weekday 1), and then a steady pickup over the week heading back to the Sunday high.  This pattern seems to be pretty stable across the five years of observations:
# MAGIC
# MAGIC **UPDATE** As part of the Spark 3 move to the [Proleptic Gregorian calendar](https://databricks.com/blog/2020/07/22/a-comprehensive-look-at-dates-and-timestamps-in-apache-spark-3-0.html), the 'u' option in CAST(DATE_FORMAT(date, 'u') was removed. We are now using 'E' to provide us a similar output.

# COMMAND ----------

# DBTITLE 1,View Weekday Trends
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   YEAR(date) AS year,
# MAGIC   (
# MAGIC     CASE
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sun' THEN 0
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Mon' THEN 1
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Tue' THEN 2
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Wed' THEN 3
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Thu' THEN 4
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Fri' THEN 5
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sat' THEN 6
# MAGIC     END
# MAGIC   ) % 7 AS weekday,
# MAGIC   AVG(sales) AS sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) AS sales
# MAGIC   FROM tmp_kaggle_train
# MAGIC   GROUP BY date
# MAGIC ) x
# MAGIC GROUP BY year, weekday
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

# MAGIC %md Now that we are oriented to the basic patterns within our data, let's explore how we might build a forecast.

# COMMAND ----------

# MAGIC %md ## Step 2: Build a Single Forecast
# MAGIC
# MAGIC Before attempting to generate forecasts for individual combinations of stores and items, it might be helpful to build a single forecast for no other reason than to orient ourselves to the use of prophet.
# MAGIC
# MAGIC Our first step is to assemble the historical dataset on which we will train the model:

# COMMAND ----------

# DBTITLE 1,Retrieve Data for a Single Item-Store Combination
# Query to aggregate data to date (ds) level.
# For prophet, the dataframe must have columns "ds" and "y" with the dates and values respectively.
sql_statement = f"""
  SELECT
    date AS ds,
    sales AS y
  FROM {temporary_training_view}
  WHERE store = 1 AND item = 1
  ORDER BY ds
"""

# assemble dataset in Pandas dataframe
history_pd = spark.sql(sql_statement).toPandas()

# drop any missing records
history_pd = history_pd.dropna()

# COMMAND ----------

# MAGIC %md Now, we will import the prophet library, but because it can be a bit verbose when in use, we will need to fine-tune the logging settings in our environment:

# COMMAND ----------

# DBTITLE 1,Import Prophet Library
# disable informational messages from prophet
logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md Based on our review of the data, it looks like we should set our overall growth pattern to linear and enable the evaluation of weekly and yearly seasonal patterns. We might also wish to set our seasonality mode to multiplicative as the seasonal pattern seems to grow with overall growth in sales:

# COMMAND ----------

# DBTITLE 1,Train Prophet Model
# set model parameters
model = prophet.Prophet(
  interval_width=0.95,
  growth="linear",
  daily_seasonality=False,
  weekly_seasonality=True,
  yearly_seasonality=True,
  seasonality_mode="multiplicative"
)

# fit the model to historical data
model.fit(history_pd)

# COMMAND ----------

# MAGIC %md Now that we have a trained model, let's use it to build a 90-day forecast:

# COMMAND ----------

# DBTITLE 1,Build Forecast
# define a dataset including both historical dates & 90-days beyond the last available date
future_pd = model.make_future_dataframe(
  freq="d", 
  periods=90, 
  include_history=True
)

# predict over the dataset
forecast_pd = model.predict(future_pd)

display(forecast_pd)

# COMMAND ----------

# MAGIC %md How did our model perform? Here we can see the general and seasonal trends in our model presented as graphs:

# COMMAND ----------

# DBTITLE 1,Examine Forecast Components
trends_fig = model.plot_components(forecast_pd)
display(trends_fig)

# COMMAND ----------

# MAGIC %md And here, we can see how our actual and predicted data line up as well as a forecast for the future, though we will limit our graph to the last year of historical data just to keep it readable:

# COMMAND ----------

# DBTITLE 1,View Historicals vs. Predictions
predict_fig = model.plot(forecast_pd, xlabel="date", ylabel="sales")

# adjust figure to display dates from last year + the 90 day forecast
xlim = predict_fig.axes[0].get_xlim()
new_xlim = (xlim[1] - (180.0 + 365.0), xlim[1] - 90.0)
predict_fig.axes[0].set_xlim(new_xlim)

display(predict_fig)

# COMMAND ----------

# MAGIC %md **NOTE** This visualization is a bit busy. Bartosz Mikulski provides [an excellent breakdown](https://www.mikulskibartosz.name/prophet-plot-explained/) of it that is well worth checking out.  In a nutshell, the black dots represent our actuals with the darker blue line representing our predictions and the lighter blue band representing our (95%) uncertainty interval.

# COMMAND ----------

# MAGIC %md Visual inspection is useful, but a better way to evaluate the forecast is to calculate Mean Absolute Error, Mean Squared Error and Root Mean Squared Error values for the predicted relative to the actual values in our set:
# MAGIC
# MAGIC **UPDATE** A change in pandas functionality requires us to use *pd.to_datetime* to coerce the date string into the right data type.

# COMMAND ----------

# DBTITLE 1,Calculate Evaluation metrics
# get historical actuals & predictions for comparison
actuals_pd = history_pd[history_pd["ds"] < datetime.date(2018, 1, 1)]["y"]
predicted_pd = forecast_pd[forecast_pd["ds"] < pd.to_datetime("2018-01-01")]["yhat"]

# calculate evaluation metrics
mae = metrics.mean_absolute_error(actuals_pd, predicted_pd)
mse = metrics.mean_squared_error(actuals_pd, predicted_pd)
rmse = math.sqrt(mse)

# print metrics to the screen
print("\n".join(["MAE: {0}", "MSE: {1}", "RMSE: {2}"]).format(mae, mse, rmse))

# COMMAND ----------

# MAGIC %md prophet provides [additional means](https://facebook.github.io/prophet/docs/diagnostics.html) for evaluating how your forecasts hold up over time. You're strongly encouraged to consider using these and those additional techniques when building your forecast models but we'll skip this here to focus on the scaling challenge.

# COMMAND ----------

# MAGIC %md ## Step 3: Scale Forecast Generation
# MAGIC
# MAGIC With the mechanics under our belt, let's now tackle our original goal of building numerous, fine-grain models & forecasts for individual store and item combinations.  We will start by assembling sales data at the store-item-date level of granularity:
# MAGIC
# MAGIC **NOTE**: The data in this data set should already be aggregated at this level of granularity but we are explicitly aggregating in the Silver Table to ensure we have the expected data structure.

# COMMAND ----------

# DBTITLE 1,Persist Data for All Store-Item Combinations in the Silver Layer
sql_statement = f"""
  SELECT
    store,
    item,
    date,
    SUM(sales) AS sales
  FROM {temporary_training_view}
  GROUP BY store, item, date
  ORDER BY store, item, date
"""

spark \
  .sql(sql_statement) \
  .write \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable(silver_table)

# COMMAND ----------

# DBTITLE 1,Retrieve Data for All Store-Item Combinations
sql_statement = f"""
  SELECT
    store,
    item,
    date AS ds,
    sales AS y
  FROM {silver_table}
"""

store_item_history = spark \
  .sql(sql_statement) \
  .repartition(sc.defaultParallelism, ["store", "item"]) \
  .cache()

# COMMAND ----------

# MAGIC %md With our data aggregated at the store-item-date level, we need to consider how we will pass our data to prophet. If our goal is to build a model for each store and item combination, we will need to pass in a store-item subset from the dataset we just assembled, train a model on that subset, and receive a store-item forecast back. We'd expect that forecast to be returned as a dataset with a structure like this where we retain the store and item identifiers for which the forecast was assembled and we limit the output to just the relevant subset of fields generated by the Prophet model:

# COMMAND ----------

# DBTITLE 1,Define Schema for Forecast Output
result_schema = types.StructType([
  types.StructField("ds", types.DateType()),
  types.StructField("store", types.IntegerType()),
  types.StructField("item", types.IntegerType()),
  types.StructField("y", types.FloatType()),
  types.StructField("yhat", types.FloatType()),
  types.StructField("yhat_upper", types.FloatType()),
  types.StructField("yhat_lower", types.FloatType())
])

# COMMAND ----------

# MAGIC %md To train the model and generate a forecast we will leverage a Pandas function.  We will define this function to receive a subset of data organized around a store and item combination.  It will return a forecast in the format identified in the previous cell:
# MAGIC
# MAGIC **UPDATE** With Spark 3.0, pandas functions replace the functionality found in pandas UDFs.  The deprecated pandas UDF syntax is still supported but will be phased out over time.  For more information on the new, streamlined pandas functions API, please refer to [this document](https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html).

# COMMAND ----------

# DBTITLE 1,Define Function to Train Model & Generate Forecast
def forecast_store_item(history_pd: pd.DataFrame) -> pd.DataFrame:
  
  # TRAIN MODEL AS BEFORE
  # --------------------------------------
  # remove missing values (more likely at day-store-item level)
  history_pd = history_pd.dropna()
  
  # configure the model
  model = prophet.Prophet(
    interval_width=0.95,
    growth="linear",
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode="multiplicative"
  )
  
  # train the model
  model.fit(history_pd)
  # --------------------------------------
  
  # BUILD FORECAST AS BEFORE
  # --------------------------------------
  # make predictions
  future_pd = model.make_future_dataframe(
    freq="d", 
    periods=90, 
    include_history=True
  )
  forecast_pd = model.predict(future_pd)  
  # --------------------------------------
  
  # ASSEMBLE EXPECTED RESULT SET
  # --------------------------------------
  # get relevant fields from forecast
  f_pd = forecast_pd[["ds", "yhat", "yhat_upper", "yhat_lower"]].set_index("ds")
  
  # get relevant fields from history
  h_pd = history_pd[["ds", "store", "item", "y"]].set_index("ds")
  
  # join history and forecast
  results_pd = f_pd.join(h_pd, how="left")
  results_pd.reset_index(level=0, inplace=True)
  
  # get store & item from incoming data set
  results_pd["store"] = history_pd["store"].iloc[0]
  results_pd["item"] = history_pd["item"].iloc[0]
  # --------------------------------------
  
  # return expected dataset
  return results_pd[["ds", "store", "item", "y", "yhat", "yhat_upper", "yhat_lower"]]  

# COMMAND ----------

# MAGIC %md There's a lot taking place within our function, but if you compare the first two blocks of code within which the model is being trained and a forecast is being built to the cells in the previous portion of this notebook, you'll see the code is pretty much the same as before. It's only in the assembly of the required result set that truly new code is being introduced and it consists of fairly standard Pandas dataframe manipulations.

# COMMAND ----------

# MAGIC %md Now let's call our pandas function to build our forecasts.  We do this by grouping our historical dataset around store and item.  We then apply our function to each group and tack on today's date as our *training_date* for data management purposes:
# MAGIC
# MAGIC **UPDATE** Per the previous update note, we are now using applyInPandas() to call a pandas function instead of a pandas UDF.

# COMMAND ----------

# DBTITLE 1,Apply Forecast Function to Each Store-Item Combination
results = store_item_history \
  .groupBy("store", "item") \
  .applyInPandas(forecast_store_item, schema=result_schema) \
  .withColumn("training_date", functions.current_date())

results.createOrReplaceTempView("tmp_new_forecasts")

display(results)

# COMMAND ----------

# MAGIC %md We we are likely wanting to report on our forecasts, so let's save them to a queryable table structure in the data lake gold layer:

# COMMAND ----------

# DBTITLE 1,Create the `forecast` Table
# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS `fgdf_accelerator_dev`.`gold`.`store_item_forecasts` (
# MAGIC   date DATE,
# MAGIC   store INTEGER,
# MAGIC   item INTEGER,
# MAGIC   sales DECIMAL(25, 18),
# MAGIC   sales_predicted DECIMAL(25, 18),
# MAGIC   sales_predicted_upper DECIMAL(25, 18),
# MAGIC   sales_predicted_lower DECIMAL(25, 18),
# MAGIC   training_date DATE
# MAGIC )
# MAGIC USING delta
# MAGIC PARTITIONED BY (date);

# COMMAND ----------

# DBTITLE 1,Persist Forecast Output
# MAGIC %sql
# MAGIC
# MAGIC -- load data to the forecast table.
# MAGIC MERGE INTO `fgdf_accelerator_dev`.`gold`.`store_item_forecasts` f
# MAGIC USING tmp_new_forecasts n 
# MAGIC ON f.date = n.ds AND f.store = n.store AND f.item = n.item
# MAGIC WHEN MATCHED THEN UPDATE SET f.date = n.ds,
# MAGIC   f.store = n.store,
# MAGIC   f.item = n.item,
# MAGIC   f.sales = n.y,
# MAGIC   f.sales_predicted = n.yhat,
# MAGIC   f.sales_predicted_upper = n.yhat_upper,
# MAGIC   f.sales_predicted_lower = n.yhat_lower,
# MAGIC   f.training_date = n.training_date
# MAGIC WHEN NOT MATCHED THEN INSERT (
# MAGIC   date,
# MAGIC   store,
# MAGIC   item,
# MAGIC   sales,
# MAGIC   sales_predicted,
# MAGIC   sales_predicted_upper,
# MAGIC   sales_predicted_lower,
# MAGIC   training_date
# MAGIC )
# MAGIC VALUES (
# MAGIC   n.ds,
# MAGIC   n.store,
# MAGIC   n.item,
# MAGIC   n.y,
# MAGIC   n.yhat,
# MAGIC   n.yhat_upper,
# MAGIC   n.yhat_lower,
# MAGIC   n.training_date
# MAGIC )

# COMMAND ----------

# MAGIC %md But how good (or bad) is each forecast?  Using the pandas function technique, we can generate evaluation metrics for each store-item forecast as follows:

# COMMAND ----------

# DBTITLE 1,Apply Same Techniques to Evaluate Each Forecast
# schema of expected result set
eval_schema = types.StructType([
  types.StructField("training_date", types.DateType()),
  types.StructField("store", types.IntegerType()),
  types.StructField("item", types.IntegerType()),
  types.StructField("mae", types.FloatType()),
  types.StructField("mse", types.FloatType()),
  types.StructField("rmse", types.FloatType())
])

# define function to calculate metrics
def evaluate_forecast(evaluation_pd: pd.DataFrame) -> pd.DataFrame:
  
  # get store & item in incoming data set
  training_date = evaluation_pd["training_date"].iloc[0]
  store = evaluation_pd["store"].iloc[0]
  item = evaluation_pd["item"].iloc[0]
  
  # calculate evaluation metrics
  mae = metrics.mean_absolute_error(evaluation_pd["y"], evaluation_pd["yhat"])
  mse = metrics.mean_squared_error(evaluation_pd["y"], evaluation_pd["yhat"])
  rmse = math.sqrt(mse)
  
  # assemble result set
  results = {
    "training_date": [training_date],
    "store": [store],
    "item": [item],
    "mae": [mae],
    "mse": [mse],
    "rmse": [rmse]
  }
  return pd.DataFrame.from_dict(results)

# calculate metrics
# - filter() limits evaluation to periods where we have historical data
results = spark \
  .table("tmp_new_forecasts") \
  .filter(functions.col("ds") < "2018-01-01") \
  .select("training_date", "store", "item", "y", "yhat") \
  .groupBy("training_date", "store", "item") \
  .applyInPandas(evaluate_forecast, schema=eval_schema)

results.createOrReplaceTempView("tmp_new_forecast_evals")

# COMMAND ----------

# MAGIC %md Once again, we will likely want to report the metrics for each forecast, so we persist these to a queryable table:

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS `fgdf_accelerator_dev`.`gold`.`store_item_forecast_evals` (
# MAGIC   store INTEGER,
# MAGIC   item INTEGER,
# MAGIC   mae DECIMAL(25, 18),
# MAGIC   mse DECIMAL(25, 18),
# MAGIC   rmse DECIMAL(25, 18),
# MAGIC   training_date DATE
# MAGIC )
# MAGIC USING delta
# MAGIC PARTITIONED BY (training_date);

# COMMAND ----------

# DBTITLE 1,Persist Evaluation Metrics
# MAGIC %sql
# MAGIC
# MAGIC -- load data to it.
# MAGIC INSERT INTO `fgdf_accelerator_dev`.`gold`.`store_item_forecast_evals`
# MAGIC SELECT
# MAGIC   store,
# MAGIC   item,
# MAGIC   mae,
# MAGIC   mse,
# MAGIC   rmse,
# MAGIC   training_date
# MAGIC FROM tmp_new_forecast_evals;

# COMMAND ----------

# MAGIC %md We now have constructed a forecast for each store-item combination and generated basic evaluation metrics for each.  To see this forecast data, we can issue a simple query (limited here to product 1 across stores 1 through 3):

# COMMAND ----------

# DBTITLE 1,Visualize Forecasts
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   store,
# MAGIC   date,
# MAGIC   item,
# MAGIC   sales_predicted,
# MAGIC   sales_predicted_upper,
# MAGIC   sales_predicted_lower
# MAGIC FROM `fgdf_accelerator_dev`.`gold`.`store_item_forecasts` a
# MAGIC WHERE item = 1 AND
# MAGIC       store IN (1, 2, 3) AND
# MAGIC       date >= '2018-01-01' AND
# MAGIC       training_date=CURRENT_DATE()
# MAGIC ORDER BY store, date, item

# COMMAND ----------

# MAGIC %md And for each of these, we can retrieve a measure of help us assess the reliability of each forecast:

# COMMAND ----------

# DBTITLE 1,Retrieve Evaluation Metrics
# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   store,
# MAGIC   mae,
# MAGIC   mse,
# MAGIC   rmse
# MAGIC FROM `fgdf_accelerator_dev`.`gold`.`store_item_forecast_evals` a
# MAGIC WHERE item = 1 AND
# MAGIC       training_date=CURRENT_DATE()
# MAGIC ORDER BY store

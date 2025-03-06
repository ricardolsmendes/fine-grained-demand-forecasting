# Databricks notebook source
# MAGIC %md The purpose of this notebook is to download and set up the data we will use for the solution accelerator. Before running this notebook, make sure you have entered your own credentials for Kaggle.

# COMMAND ----------

# MAGIC %pip install kaggle

# COMMAND ----------

# MAGIC %md 
# MAGIC Set Kaggle credential configuration values in the block below: You can set up a [secret scope](https://docs.databricks.com/security/secrets/secret-scopes.html) to manage credentials used in notebooks. For the block below, we have set up the `fgdf-accelerator-dev` secret scope using Terraform and saved our credentials there for internal testing purposes.

# COMMAND ----------

import os

secret_scope = "fgdf-accelerator-dev"

# os.environ["KAGGLE_USERNAME"] = "YOUR KAGGLE USERNAME HERE" # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ["KAGGLE_USERNAME"] = dbutils.secrets.get(secret_scope, "kaggle-username")

# os.environ["KAGGLE_KEY"] = "YOUR KAGGLE KEY HERE" # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ["KAGGLE_KEY"] = dbutils.secrets.get(secret_scope, "kaggle-key")

# COMMAND ----------

# MAGIC %md Download the data from Kaggle using the credentials set above:

# COMMAND ----------

# MAGIC %sh 
# MAGIC
# MAGIC # The /databricks/driver directory is typically used to store files that are accessible to the driver node in a Databricks cluster.
# MAGIC # This is a good place to download and unzip files that will be used in Databricks notebooks.
# MAGIC cd /databricks/driver
# MAGIC
# MAGIC # The Kaggle client uses the environment variables set in the previous step to authenticate.
# MAGIC kaggle competitions download -c demand-forecasting-kernels-only
# MAGIC unzip -o demand-forecasting-kernels-only.zip

# COMMAND ----------

# MAGIC %md Move the downloaded data to the bronze layer used throughout the accelerator:

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE VOLUME IF NOT EXISTS `fgdf_accelerator_dev`.`bronze`.`kaggle`;

# COMMAND ----------

downloaded_file_path = "/databricks/driver/train.csv"

catalog = "fgdf_accelerator_dev"
landing_file_path = f"/Volumes/{catalog}/bronze/kaggle/train.csv"

dbutils.fs.cp(f"file:{downloaded_file_path}", landing_file_path)

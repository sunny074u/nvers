# Databricks notebook source
import unittest
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
from delta.tables import DeltaTable
from pyspark.sql.functions import current_date
import importlib
import sys
import os

class TestDataTransformation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize a Spark session for the test class with Delta support
        cls.spark = SparkSession.builder \
            .appName("PySpark Unit Test") \
            .master("local[*]") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        
        cls.spark.conf.set("spark.sql.shuffle.partitions", "1")
    
    @patch('sys.path.append')
    @patch('importlib.import_module')
    def test_paths_and_modules_imports(self, mock_import_module, mock_sys_path_append):
        # Mock the import of modules dynamically
        usr_path = '/Workspace/Users/sun_edisi@outlook.com/nvers/src'
        paths_and_modules = {
            f'{usr_path}/common/schemas/': ['silver.company_profile'],
            f'{usr_path}/common/': ['utils'],
            f'{usr_path}/common/schemas/': ['silver.util_func'],
        }
        
        for path, modules in paths_and_modules.items():
            abs_path = os.path.abspath(path)
            if abs_path not in sys.path:
                sys.path.append(abs_path)
                mock_sys_path_append.assert_called_with(abs_path)  # Ensure sys.path.append was called
            
            for module in modules:
                # Mocking the import of each module
                globals()[module] = importlib.import_module(module)
                mock_import_module.assert_called_with(module)  # Ensure importlib.import_module was called correctly

    @patch('silver.util_func.load_bronze_data')
    @patch('silver.util_func.read_delta_to_df')
    def test_data_transformation(self, mock_read_delta, mock_load_bronze_data):
        # Mock the reading of Delta data
        symbol_mapping_data = [("MSFT", 1), ("AAPL", 2)]
        symbol_mapping_df = self.spark.createDataFrame(symbol_mapping_data, ["Symbol", "CompanyId"])
        mock_read_delta.return_value = symbol_mapping_df

        # Mock the bronze data
        bronze_data = [
            ("MSFT", "2024-03-05T14:30:00", 100.0, 110.0, 90.0, 105.0, 1000, 202403, 14),
            ("AAPL", "2024-03-05T14:30:00", 200.0, 220.0, 180.0, 210.0, 2000, 202403, 14),
        ]
        mock_bronze_df = self.spark.createDataFrame(bronze_data, ["symbol", "timestamp", "open", "high", "low", "close", "volume", "TradeYearMonth", "Hour"])
        mock_load_bronze_data.return_value = mock_bronze_df

        # Perform the data transformation
        bronze_df = mock_bronze_df.selectExpr(
            "symbol as Symbol",
            "cast(timestamp as Timestamp) as TradeDate",
            "CAST(open as DOUBLE) as Open",
            "CAST(high as DOUBLE) as High",
            "CAST(low as DOUBLE) as Low",
            "CAST(close as DOUBLE) as Close",
            "CAST(volume as BIGINT) as Volume",
            "TradeYearMonth",
            "Hour"
        )
        
        company_profile_df = bronze_df.join(symbol_mapping_df, on="Symbol", how="left").select(
            "CompanyId",
            "Symbol",
            "TradeDate",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "TradeYearMonth",
            "Hour"
        ).withColumn("EffectiveDate", current_date())

        # Validate the transformation
        expected_data = [
            (1, "MSFT", "2024-03-05T14:30:00", 100.0, 110.0, 90.0, 105.0, 1000, 202403, 14, current_date()),
            (2, "AAPL", "2024-03-05T14:30:00", 200.0, 220.0, 180.0, 210.0, 2000, 202403, 14, current_date()),
        ]
        expected_df = self.spark.createDataFrame(expected_data, company_profile_df.schema)

        self.assertEqual(company_profile_df.collect(), expected_df.collect())

if __name__ == '__main__':
    unittest.main()


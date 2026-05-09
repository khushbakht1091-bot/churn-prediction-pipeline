import great_expectations as gx
import pandas as pd
from pathlib import Path


# Define project root and data path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn_cleaned.csv"

# Load the data
df = pd.read_csv(DATA_PATH)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Create a DataContext (in-memory)
context = gx.get_context()

# Create an ExpectationSuite (our named checklist)
suite = gx.ExpectationSuite(name="churn_data_suite")
suite = context.suites.add(suite)

print("ExpectationSuite created successfully")

# Column existence checks 
suite.add_expectation(gx.expectations.ExpectColumnToExist(column="tenure"))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column="MonthlyCharges"))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Contract"))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Churn"))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column="customerID"))

print("Column existence checks added")

# Null checks 
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="tenure"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Contract"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Churn"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="customerID"))

print("Null checks added")

# Range checks 
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="tenure",
    min_value=0,
    max_value=72
))

suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="MonthlyCharges",
    min_value=0,
    max_value=150
))

print("Range checks added")

# Allowed values checks 
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
    column="Contract",
    value_set=["Month-to-month", "One year", "Two year"]
))

suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
    column="Churn",
    value_set=["Yes", "No"]
))

suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
    column="gender",
    value_set=["Male", "Female"]
))

suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
    column="InternetService",
    value_set=["DSL", "Fiber optic", "No"]
))

print("Allowed values checks added")


# VALIDATION 
# Create a data source from our pandas dataframe
data_source = context.data_sources.add_pandas("churn_pandas_source")

# Create a data asset (a named reference to our dataframe)
data_asset = data_source.add_dataframe_asset(name="churn_dataframe_asset")

# Build a batch definition
batch_definition = data_asset.add_batch_definition_whole_dataframe("churn_batch")

# Create and register the validation definition
validation_definition = gx.ValidationDefinition(
    name="churn_validation",
    data=batch_definition,
    suite=suite
)
validation_definition = context.validation_definitions.add(validation_definition)

# Run validation
results = validation_definition.run(batch_parameters={"dataframe": df})

print(results)

# RESULTS SUMMARY 
stats = results.statistics
print(f"\nValidation Results")
print(f"Total expectations: {stats['evaluated_expectations']}")
print(f"Passed: {stats['successful_expectations']}")
print(f"Failed: {stats['unsuccessful_expectations']}")
print(f"Success rate: {stats['success_percent']}%")

# HARD STOP IF VALIDATION FAILS 
if not results.success:
    failed = [
        r.expectation_config.type
        for r in results.results
        if not r.success
    ]
    raise ValueError(f"Data validation failed. Failed expectations: {failed}")

print("\nData validation passed. Safe to proceed.")
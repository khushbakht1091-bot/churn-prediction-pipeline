import great_expectations as gx
import pandas as pd
from pathlib import Path


def validate_data(df: pd.DataFrame) -> None:
    """
    Run Great Expectations validation on the input DataFrame.
    Raises ValueError if any expectation fails.
    """
    # Create a DataContext (in-memory)
    context = gx.get_context()

    # Create an ExpectationSuite
    suite = gx.ExpectationSuite(name="churn_data_suite")
    suite = context.suites.add(suite)

    # Column existence checks
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="tenure"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="MonthlyCharges"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Contract"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Churn"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="customerID"))

    # Null checks
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="tenure"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Contract"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Churn"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="customerID"))

    # Range checks
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="tenure", min_value=0, max_value=72
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="MonthlyCharges", min_value=0, max_value=150
    ))

    # Allowed values checks
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="Contract",
        value_set=["Month-to-month", "One year", "Two year"]
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="Churn", value_set=["Yes", "No"]
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="gender", value_set=["Male", "Female"]
    ))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="InternetService",
        value_set=["DSL", "Fiber optic", "No"]
    ))

    # Validation setup
    data_source = context.data_sources.add_pandas("churn_pandas_source")
    data_asset = data_source.add_dataframe_asset(name="churn_dataframe_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("churn_batch")

    validation_definition = gx.ValidationDefinition(
        name="churn_validation",
        data=batch_definition,
        suite=suite
    )
    validation_definition = context.validation_definitions.add(validation_definition)

    # Run validation
    results = validation_definition.run(batch_parameters={"dataframe": df})

    # Results summary
    stats = results.statistics
    print(f"\nValidation Results")
    print(f"Total expectations: {stats['evaluated_expectations']}")
    print(f"Passed: {stats['successful_expectations']}")
    print(f"Failed: {stats['unsuccessful_expectations']}")
    print(f"Success rate: {stats['success_percent']}%")

    # Hard stop
    if not results.success:
        failed = [
            r.expectation_config.type
            for r in results.results
            if not r.success
        ]
        raise ValueError(f"Data validation failed. Failed expectations: {failed}")

    print("\nData validation passed. Safe to proceed.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn_cleaned.csv"
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    validate_data(df)
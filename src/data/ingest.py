import os
import pandas as pd
from sqlalchemy import create_engine, text
from src.data.validate_data import validate_data


def get_engine():
    db_user = os.environ.get("CHURN_DB_USER")
    db_password = os.environ.get("CHURN_DB_PASSWORD")
    db_host = os.environ.get("CHURN_DB_HOST")
    db_port = os.environ.get("CHURN_DB_PORT")
    db_name = os.environ.get("CHURN_DB_NAME")

    if not all([db_user, db_password, db_host, db_port, db_name]):
        raise ValueError("Database credentials missing. Check your environment variables.")

    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(connection_string)


def ingest_data(csv_path: str) -> int:
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    print("Running data validation...")
    validate_data(df)
    print("Validation passed")

    engine = get_engine()

    print("Loading data into PostgreSQL...")
    df.to_sql(
        name="raw_churn_data",
        con=engine,
        if_exists="replace",
        index=False
    )

    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM raw_churn_data"))
        row_count = result.scalar()

    print(f"Successfully ingested {row_count} rows into raw_churn_data")
    return row_count


if __name__ == "__main__":
    csv_path = os.path.join("data", "raw", "telco_churn_cleaned.csv")
    row_count = ingest_data(csv_path)
    print(f"Ingestion complete. {row_count} rows in database.")
CREATE USER churn_user WITH PASSWORD 'churn_pass';
CREATE DATABASE churn_db OWNER churn_user;
GRANT ALL PRIVILEGES ON DATABASE churn_db TO churn_user;
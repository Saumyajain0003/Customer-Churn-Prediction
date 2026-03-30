"""
seed_db.py — Initializes a SQLite database and normalizes the raw CSV data into separate tables.
This simulates a real-world relational database environment for the data science pipeline.
"""
import sqlite3
import pandas as pd
import logging
from pathlib import Path
from src.config import DATA_FILE, ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = ROOT_DIR / "data" / "churn.db"

def create_database():
    """Reads the raw CSV and populates a normalized SQLite database."""
    if not DATA_FILE.exists():
        logger.error(f"Raw data file not found at {DATA_FILE}")
        return

    logger.info("Reading raw CSV data...")
    df = pd.read_csv(DATA_FILE)

    # Ensure output directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Connecting to SQLite database at {DB_PATH}")
    # This will create the db file if it doesn't exist
    conn = sqlite3.connect(DB_PATH)

    try:
        # 1. Customers Table (Demographics)
        logger.info("Creating 'customers' table...")
        customers_df = df[['CustomerId', 'Surname', 'Geography', 'Gender', 'Age']].drop_duplicates()
        customers_df.to_sql('customers', conn, if_exists='replace', index=False)

        # 2. Financials Table
        logger.info("Creating 'financials' table...")
        financials_df = df[['CustomerId', 'CreditScore', 'Balance', 'EstimatedSalary']].drop_duplicates()
        financials_df.to_sql('financials', conn, if_exists='replace', index=False)

        # 3. Activity Table
        logger.info("Creating 'activity' table...")
        activity_df = df[['CustomerId', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']].drop_duplicates()
        activity_df.to_sql('activity', conn, if_exists='replace', index=False)

        # 4. Churn Status Table
        logger.info("Creating 'churn_status' table...")
        status_df = df[['CustomerId', 'Exited']].drop_duplicates()
        status_df.to_sql('churn_status', conn, if_exists='replace', index=False)

        # 5. Predictions Table (Empty initially)
        logger.info("Creating empty 'customer_churn_predictions' table...")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_churn_predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER,
                churn_probability REAL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(customer_id) REFERENCES customers(CustomerId)
            )
        """)
        conn.commit()
        
        logger.info("✅ Database seeding complete! Tables created: customers, financials, activity, churn_status, customer_churn_predictions.")
        
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_database()

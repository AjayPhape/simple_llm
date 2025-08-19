import logging

import psycopg2
from simple_llm.app.config import db_settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    def __init__(self):
        self.dbname = db_settings.dbname
        self.user = db_settings.user
        self.password = db_settings.password
        self.host = db_settings.host
        self.port = db_settings.port
        self.connection = None

    def __enter__(self):
        try:
            self.connection = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
            )
            logger.info("Connection to PostgreSQL established successfully.")
            return self
        except psycopg2.Error as e:
            logger.exception(f"Error connecting to PostgreSQL: {e}")
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL connection closed.")

    def fetchone(self, query, params=None):
        with self.connection.cursor() as cursor:
            logger.info(
                cursor.mogrify(query, params).decode("utf-8")
            )  # Print the query with parameters
            cursor.execute(query, params)
            return cursor.fetchone()

    def execute(self, query, params=None):
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            self.connection.commit()

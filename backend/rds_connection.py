import psycopg2  # type:ignore
from backend.config import settings


class RelationalDatabaseConnector:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=settings.DB_HOST,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            port=settings.DB_PORT,
        )
        self.cursor = self.conn.cursor()

    def execute_fetch_query_on_rds(self, query, params=None):
        self.cursor.execute(query, params or ())
        return self.cursor.fetchall()

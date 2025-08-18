import snowflake.connector
import pandas as pd
from typing import List, Dict, Any
import logging
from config import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnowflakeConnector:
    """Handles Snowflake database connections and queries for Netflix titles data."""
    
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establish connection to Snowflake."""
        try:
            self.connection = snowflake.connector.connect(
                user=SNOWFLAKE_USER,
                password=SNOWFLAKE_PASSWORD,
                account=SNOWFLAKE_ACCOUNT,
                warehouse=SNOWFLAKE_WAREHOUSE,
                database=SNOWFLAKE_DATABASE,
                schema=SNOWFLAKE_SCHEMA,
                client_session_keep_alive=True,
                validate_default_parameters=True
            )
            logger.info("Successfully connected to Snowflake")
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            raise
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame."""
        try:
            if not self.connection or self.connection.is_closed():
                self.connect()
            
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Fetch results
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            
            cursor.close()
            
            # Convert to DataFrame
            df = pd.DataFrame(results, columns=column_names)
            
            # Validate data quality
            if not df.empty:
                # Clean column names (remove extra spaces, standardize case)
                df.columns = [col.strip().upper() for col in df.columns]
                
                # Handle missing values
                df = df.fillna('Unknown')
                
                # Ensure consistent data types
                if 'RELEASE_YEAR' in df.columns:
                    df['RELEASE_YEAR'] = pd.to_numeric(df['RELEASE_YEAR'], errors='coerce').fillna(0).astype(int)
                
                logger.info(f"Query executed successfully, returned {len(df)} rows")
            else:
                logger.info("Query executed successfully but returned no results")
            
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_table_schema(self) -> Dict[str, Any]:
        """Get the schema of the netflix_titles table."""
        query = """
        DESCRIBE TABLE netflix_titles
        """
        try:
            df = self.execute_query(query)
            schema_info = {}
            for _, row in df.iterrows():
                schema_info[row['name']] = {
                    'type': row['type'],
                    'null': row['null'],
                    'default': row['default']
                }
            return schema_info
        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            return {}
    
    def get_sample_data(self, limit: int = 5) -> pd.DataFrame:
        """Get sample data from netflix_titles table."""
        query = f"""
        SELECT * FROM netflix_titles 
        LIMIT {limit}
        """
        return self.execute_query(query)
    
    def search_titles(self, search_term: str, limit: int = 10) -> pd.DataFrame:
        """Search for titles containing the search term."""
        query = f"""
        SELECT * FROM netflix_titles 
        WHERE LOWER(title) LIKE LOWER('%{search_term}%')
        OR LOWER(description) LIKE LOWER('%{search_term}%')
        LIMIT {limit}
        """
        return self.execute_query(query)
    
    def get_titles_by_type(self, content_type: str, limit: int = 10) -> pd.DataFrame:
        """Get titles by content type (Movie, TV Show)."""
        query = f"""
        SELECT * FROM netflix_titles 
        WHERE LOWER(type) = LOWER('{content_type}')
        LIMIT {limit}
        """
        return self.execute_query(query)
    
    def get_titles_by_country(self, country: str, limit: int = 10) -> pd.DataFrame:
        """Get titles by country."""
        query = f"""
        SELECT * FROM netflix_titles 
        WHERE LOWER(country) LIKE LOWER('%{country}%')
        LIMIT {limit}
        """
        return self.execute_query(query)
    
    def get_titles_by_year(self, year: int, limit: int = 10) -> pd.DataFrame:
        """Get titles by release year."""
        query = f"""
        SELECT * FROM netflix_titles 
        WHERE release_year = {year}
        LIMIT {limit}
        """
        return self.execute_query(query)
    
    def get_titles_by_rating(self, rating: str, limit: int = 10) -> pd.DataFrame:
        """Get titles by rating."""
        query = f"""
        SELECT * FROM netflix_titles 
        WHERE LOWER(rating) = LOWER('{rating}')
        LIMIT {limit}
        """
        return self.execute_query(query)
    
    def get_titles_by_director(self, director: str, limit: int = 10) -> pd.DataFrame:
        """Get titles by director."""
        query = f"""
        SELECT * FROM netflix_titles 
        WHERE LOWER(director) LIKE LOWER('%{director}%')
        LIMIT {limit}
        """
        return self.execute_query(query)
    
    def get_titles_by_director_and_type(self, director: str, content_type: str, limit: int = 10) -> pd.DataFrame:
        """Get titles by director and content type (e.g., movies by James Cameron)."""
        query = f"""
        SELECT * FROM netflix_titles 
        WHERE LOWER(director) LIKE LOWER('%{director}%')
        AND LOWER(type) = LOWER('{content_type}')
        LIMIT {limit}
        """
        return self.execute_query(query)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the netflix_titles table."""
        queries = {
            'total_titles': "SELECT COUNT(*) as total FROM netflix_titles",
            'by_type': "SELECT type, COUNT(*) as count FROM netflix_titles GROUP BY type ORDER BY count DESC",
            'by_rating': "SELECT rating, COUNT(*) as count FROM netflix_titles GROUP BY rating ORDER BY count DESC",
            'by_year': "SELECT release_year, COUNT(*) as count FROM netflix_titles GROUP BY release_year ORDER BY release_year DESC LIMIT 15",
            'by_country': "SELECT country, COUNT(*) as count FROM netflix_titles GROUP BY country ORDER BY count DESC LIMIT 10",
            'by_genre': "SELECT listed_in, COUNT(*) as count FROM netflix_titles GROUP BY listed_in ORDER BY count DESC LIMIT 10"
        }
        
        stats = {}
        for stat_name, query in queries.items():
            try:
                df = self.execute_query(query)
                if stat_name == 'total_titles':
                    stats[stat_name] = df.iloc[0]['total']
                else:
                    stats[stat_name] = df.to_dict('records')
            except Exception as e:
                logger.error(f"Failed to get {stat_name} statistics: {e}")
                stats[stat_name] = None
        
        return stats
    
    def get_title_details(self, title_id: str = None, title: str = None) -> pd.DataFrame:
        """Get detailed information about a specific title."""
        if title_id:
            query = f"SELECT * FROM netflix_titles WHERE show_id = '{title_id}'"
        elif title:
            query = f"SELECT * FROM netflix_titles WHERE LOWER(title) = LOWER('{title}')"
        else:
            return pd.DataFrame()
        
        return self.execute_query(query)
    
    def get_similar_titles(self, title: str, limit: int = 5) -> pd.DataFrame:
        """Find titles similar to the given title based on genre and type."""
        query = f"""
        SELECT * FROM netflix_titles 
        WHERE LOWER(listed_in) IN (
            SELECT LOWER(listed_in) FROM netflix_titles 
            WHERE LOWER(title) = LOWER('{title}')
        )
        AND LOWER(title) != LOWER('{title}')
        LIMIT {limit}
        """
        return self.execute_query(query)
    
    def close(self):
        """Close the Snowflake connection."""
        if self.connection and not self.connection.is_closed():
            self.connection.close()
            logger.info("Snowflake connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

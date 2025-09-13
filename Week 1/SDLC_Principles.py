import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load CSV
def load_data(data_location: str) -> pd.DataFrame:
    """Load CSV data from a URL."""
    try:
        return pd.read_csv(data_location)
    except Exception as e:
        logger.error(f"Failed to load data from {data_location}: {str(e)}")
        raise Exception(f"Failed to load data from {data_location}: {str(e)}")

# Get average sepal length
def get_average_sepal_length(dataframe: pd.DataFrame) -> float:
    """Get the average sepal length from the dataframe."""
    if 'sepal_length' not in dataframe.columns:
        logger.error("Column 'sepal_length' not found in the dataframe")
        raise KeyError("Column 'sepal_length' not found in the dataframe")
    return dataframe['sepal_length'].mean()

# Get max petal width
def get_max_petal_width(dataframe: pd.DataFrame) -> float:
    """Get the maximum petal width from the dataframe."""
    if 'petal_width' not in dataframe.columns:
        logger.error("Column 'petal_width' not found in the dataframe")
        raise KeyError("Column 'petal_width' not found in the dataframe")
    return dataframe['petal_width'].max()

# Filter rows based on species
def filter_species(dataframe: pd.DataFrame, species: str) -> pd.DataFrame:
    """Filter the dataframe for rows where species matches the given species."""
    if 'species' not in dataframe.columns:
        logger.error("Column 'species' not found in the dataframe")
        raise KeyError("Column 'species' not found in the dataframe")
    return dataframe[dataframe['species'] == species]

if __name__ == "__main__":
    try:
        data_location = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        logger.info(f"Loading data from {data_location}")
        df = load_data(data_location)
        
        avg_sepal_length = get_average_sepal_length(df)
        logger.info(f"Average sepal length: {avg_sepal_length}")
        
        max_petal_width = get_max_petal_width(df)
        logger.info(f"Max petal width: {max_petal_width}")
        
        species = "setosa"
        filtered_df = filter_species(df, species)
        logger.info(f"Filtered data for species '{species}':\n{filtered_df}")
            
    except Exception as e:
        logger.error(f"Failed to process data: {str(e)}")

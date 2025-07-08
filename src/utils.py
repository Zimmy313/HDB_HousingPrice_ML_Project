import pandas as pd
import re 

def clean_dataset(df, dataset_name):
    """
    Cleans the dataset by identifying and removing duplicate rows and checking for missing values.

    Parameters:
    df (pd.DataFrame): The DataFrame to be cleaned.
    dataset_name (str): The name of the dataset for logging purposes.

    Returns:
    pd.DataFrame: A cleaned DataFrame with duplicates removed.
    
    Prints:
    - The number of duplicate rows found and removed.
    - The count of missing values per column.
    """
    print(f"Cleaning dataset: {dataset_name}")
    
    # Check for duplicates
    duplicates_count = df.duplicated().sum()
    print(f"{dataset_name}: Number of duplicates: {duplicates_count}")
    
    # Drop duplicates
    df_cleaned = df.drop_duplicates().copy()          
    
    # Check for missing values
    missing_values = df_cleaned.isna().sum()
    print(f"{dataset_name}: Missing values per column:")
    print(missing_values)
    
    return df_cleaned

def convert_month_to_datetime(df, column_name='month'):
    """
    Converts a column containing month information in 'YYYY-MM' format to datetime.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the month column.
    column_name (str): The name of the column with month data (default is 'Month').

    Returns:
    pd.DataFrame: The DataFrame with the specified column converted to datetime.
    """
    df[column_name] = pd.to_datetime(df[column_name], format='%Y-%m')
    return df

def clean_flat_model(df):
    """
    Cleans the 'flat_model' column in the given DataFrame by removing any 
    leading or trailing whitespace and converting the text to uppercase.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the 'flat_model' column.

    Returns:
        pd.DataFrame: The modified DataFrame with the cleaned 'flat_model' column.
    """
    df['flat_model'] = df['flat_model'].str.strip().str.upper()
    return df

def impute_remaining_lease(df):
    """
    Imputes 'remaining_lease' as a float with 1 decimal place based on 'lease_commence_date' and 'Month' columns,
    assuming a 99-year lease duration. The decimal part represents the fraction of the year based on months.

    Parameters:
    df (pd.DataFrame): The DataFrame where 'remaining_lease' needs to be imputed.

    Returns:
    pd.DataFrame: The DataFrame with 'remaining_lease' imputed as a float to 1 decimal place.
    """
    # Extract the current year and month from 'Month' column
    df['current_year'] = df['month'].dt.year
    df['current_month'] = df['month'].dt.month

    # Calculate the number of years and fraction of the year remaining
    years_elapsed = df['current_year'] - df['lease_commence_date']
    months_fraction = df['current_month'] / 12  # Convert months to fraction of a year

    # Calculate remaining lease as a float with 1 decimal place
    df['remaining_lease'] = (99 - years_elapsed - months_fraction).round(1)

    # Drop the temporary columns
    df.drop(columns=['current_year', 'current_month'], inplace=True)
    return df


def standardize_remaining_lease(df):
    """
    Standardizes the 'remaining_lease' column in a DataFrame to a float with 1 decimal place.
    Converts from 'XX years YY months' or 'XX years' format to a float where the decimal represents the fraction of the year.
    Caps any 'remaining_lease' value at 99 years.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'remaining_lease' column in mixed formats.

    Returns:
    pd.DataFrame: The DataFrame with 'remaining_lease' standardized as a float with 1 decimal place and capped at 99 years.
    """
    def convert_lease(lease):
        # If the value is already a float or int, return it as-is, capped at 99
        if isinstance(lease, (float, int)):
            return min(round(float(lease), 1), 99.0)
        
        # Handle 'XX years YY months' format
        match_full = re.match(r"(\d+) years (\d+) months", lease)
        if match_full:
            years = int(match_full.group(1))
            months = int(match_full.group(2))
            remaining_lease = years + (months / 12)
            return min(round(remaining_lease, 1), 99.0)
        
        # Handle 'XX years' format only
        match_years = re.match(r"(\d+) years", lease)
        if match_years:
            years = int(match_years.group(1))
            return min(round(float(years), 1), 99.0)
        
        # Return None if the format is unexpected
        return None
    
    # Apply the conversion and capping directly to the 'remaining_lease' column
    df['remaining_lease'] = df['remaining_lease'].apply(convert_lease)
    return df

def split_categorical_numerical(df):
    """
    Splits the DataFrame into separate categorical and numerical DataFrames based on column data types.

    Parameters:
    df (pd.DataFrame): The DataFrame to be split.

    Returns:
    tuple: A tuple containing two DataFrames:
        - categorical_columns (pd.DataFrame): DataFrame of categorical columns.
        - numerical_columns (pd.DataFrame): DataFrame of numerical columns, including datetime columns.
    """
    categorical_columns = df.select_dtypes(include=['object'])
    numerical_columns = df.select_dtypes(include=['number', 'datetime64[ns]'])
    return categorical_columns, numerical_columns

def split_storey_range(df):
    """
    Splits the 'storey_range' column into 'lower_storey' and 'upper_storey'.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame containing the 'storey_range' column.
    
    Returns:
    pd.DataFrame: Modified DataFrame with 'lower_storey' and 'upper_storey' columns added.
    """
    # Splitting the 'storey_range' column into 'lower_storey' and 'upper_storey'
    df[['lower_storey', 'upper_storey']] = df['storey_range'].str.split(' TO ', expand=True)

    # Convert lower_storey and upper_storey to numeric types
    df['lower_storey'] = pd.to_numeric(df['lower_storey'], errors='coerce')
    df['upper_storey'] = pd.to_numeric(df['upper_storey'], errors='coerce')

    return df

def add_max_storey(df):
    """
    Adds a column 'max_storey' to the DataFrame based on the maximum upper_storey 
    for each unique combination of town, block, and street_name.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing 'town', 'block', 'street_name', and 'upper_storey'.

    Returns:
    pd.DataFrame: Modified DataFrame with the new 'max_storey' column.
    """
    
    # Create max_storey column
    df['max_storey'] = df.groupby(['town', 'block', 'street_name'])['upper_storey'].transform('max')
    
    return df

def calculate_age_of_flat(row):
    """
Calculates the 'age_of_flat' for each row in the DataFrame based on the 'remaining_lease'

Parameters:
row (pd.Series): A row of the DataFrame containing 'remaining_lease', 'year',
                 'lease_commence_date', and 'month_number'.

Returns:
float: The calculated 'age_of_flat', which is set to 0 if the result is negative.
"""

    age = 99 - row['remaining_lease']
    
    return max(age, 0)
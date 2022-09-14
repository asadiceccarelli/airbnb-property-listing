import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)


def remove_rows_with_missing_ratings(df):
    """Removes rows where data is missing in ratings column.
    Args:
        df (DataFrame): DataFrame with missing ratings.
    Returns:
        df (DataFrame): DataFrame with rows containing missing ratings removed.
    """
    logging.info('Removing rows with missing ratings...')
    df.dropna(subset=['cleanliness_rate', 'accuracy_rate', 'communication_rate', 'location_rate'], inplace=True)
    return df


def combine_description_strings(df):
    """Concatenates column with lists into a string.
    Args:
        df (DataFrame): DataFrame with lists in column.
    Returns:
        df (DataFrame): DataFrame with rows with only strings in description column.
    """
    logging.info('Concatenating lists in column...')
    df['description'] = df.description.apply(lambda x: ' '.join(x) if type(x)==list else x)
    return df


def set_default_feature_values(df):
    """Sets default value of guests, beds, bathrooms and bedrooms.
    Args:
        df (DataFrame): DataFrame with missing data.
    Returns:
        df (DataFrame): DataFrame with missing data values set to 1.
    """
    logging.info('Setting default values...')
    df[['guests', 'beds', 'bathrooms', 'bedrooms']] = df[['guests', 'beds', 'bathrooms', 'bedrooms']].fillna(1)
    return df


def clean_data(df):
    """Removes rows where ratings are missing,
        Concatenates description column and set default
        value for guests, beds, bathrooms and bedrooms.
    Args:
        df (DataFrame): DataFrame to be cleaned.
    Returns:
        df (DataFrame): Cleaned DataFrame.
    """
    logging.info('Cleaning data...')
    df.columns = df.columns.str.lower()
    remove_rows_with_missing_ratings(df)
    combine_description_strings(df)
    set_default_feature_values(df)
    return df


def load_airbnb(df, labels):
    """Splits the DataFrame into features and labels, ready
        to train a model.
    Args:
        df (DataFrame): Complete DataFrame.
        labels (list): Column names of labels
    Returns:
        (tuple): Tuple containing features and labels.
    """
    logging.info('Splitting into features and labels...')
    labels_df = df[labels]
    features_df = df.drop(labels, axis=1)
    return (features_df, labels_df)


def create_numerical_dataset(df):
    """Drops all non-numerical values from DataFrame.
    Args:
        df (DataFrame): Original DataFrame.
    Returns:
        (DataFrame): DataFrame only containing numerical values.
    """
    logging.info('Creating numerical dataset...')
    return df[[
        'guests', 'beds', 'bathrooms', 'price_night', 'cleanliness_rate',
        'accuracy_rate', 'communication_rate', 'location_rate', 'check-in_rate',
        'value_rate', 'amenities_count', 'bedrooms'
        ]]

if __name__ == '__main__':
    airbnb_data = pd.read_csv('dataframes/cleaned_dataset.csv')
    create_numerical_dataset(airbnb_data).to_csv('dataframes/numerical_data.csv')
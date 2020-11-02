#import statements
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories from csv and merges them
    :param messages_filepath: (string) filepath to messages csv
    :param categories_filepath: (string) filepath to categories csv
    :return: (pandas dataframe) merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the datasets
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    """
    Expands the message categories to one column per category. Cleans the data so the category rows only contain
    1 or 0. Removes duplicates.
    :param df: (pandas dataframe) The loaded and merges dataframe of messages and categories
    :return: (pandas dataframe) Cleaned and transformed
    """
    # creates a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    # selects the first row of the categories dataframe
    row = categories.iloc[0]
    # extract the column names
    category_colnames = row.apply(lambda x: x[:-2])
    # renames the columns of `categories`
    categories.columns = category_colnames

    # iterates through columns to reduce the entries to 1 or 0
    for column in categories:
        # sets each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # sets value of "2" to the mode
        column_mode = categories[column].mode()[0]
        categories[column] = categories[column].apply(lambda x: x.replace("2", column_mode))
        # converts column from string to numeric
        categories[column] = categories[column].astype(int)

    # drops the original categories column from `df`
    df.drop(columns="categories", inplace=True)
    # concatenates the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drops duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    Loads the dataframe from the database with the messages and labeled categories.
    :param database_filepath: (string) path to database file
    :return: X: messages, Y: labeled categories, category names
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine, index_col="id")
    X = df["message"]
    Y = df.iloc[:, 3:]

    return X, Y, Y.columns


def tokenize(text):
    """
    Gets handed a string and returns it as lemmatized tokens.
    :param text: (string) one message, of the messages
    :return: message as list of tokens
    """
    finished_tokens = []
    tokens = word_tokenize(text)
    lemma = WordNetLemmatizer()

    for token in tokens:
        processed_token = lemma.lemmatize(token).lower().strip()
        finished_tokens.append(processed_token)

    return finished_tokens


def build_model():
    """
    Builds a pipeline with the three components Countvectorizer, Tfidf transformer and a
    Random Forest as multi output classifier
    :return: the pipeline object
    """
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("moclf", MultiOutputClassifier(RandomForestClassifier()))
    ])
    # parameters for gridsearch
    parameters = {
        'moclf__estimator__n_estimators': [50, 100],
        'vect__max_df': [0.5, 1.0],
    }

    grid = GridSearchCV(pipeline, param_grid=parameters)

    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predicts the Y values from X_test and prints a classification report for each category.
    :param model: (pipeline object) classification pipeline model
    :param X_test: (vector shaped (n,)) X_test values the model hasn't seen yet
    :param Y_test: (matrix shaped(n,36)) The actual outcomes for the X_test values
    :param category_names: (list of strings) The category names of the Y values
    :return: prints the classification report
    """
    y_pred = model.predict(X_test)

    for idx, name in enumerate(category_names):
        print("Score for {}:".format(name))
        print(classification_report(Y_test.iloc[:, idx], y_pred[:, idx]))


def save_model(model, model_filepath):
    """
    Saves the classification model at the given filepath
    :param model: (pipeline object) The classification model
    :param model_filepath: (string) filepath
    :return: None
    """
    # Dump the trained decision tree classifier with Pickle
    model_filename = model_filepath
    # Open the file to save as pkl file
    cv_model_pkl = open(model_filename, 'wb')
    pickle.dump(model, cv_model_pkl)
    # Close the pickle instances
    cv_model_pkl.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

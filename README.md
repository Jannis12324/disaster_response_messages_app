# Disaster Response Pipeline Project
This project creates a web app with flask and designs one ETL pipeline for the
messages and one machine learning pipeline to train a model to classify the
messages. The model is loaded into the webapp and is used to classify new incoming
messages.
# Date
Created on the 29.10.2020
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
# Data
Labeled messages from disaster were provided by the company [figure eight](https://f8-federal.com/)

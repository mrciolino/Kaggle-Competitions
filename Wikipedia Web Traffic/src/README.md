# Wikipedia-Traffic-Time-Series Sources Files

Download Data
* Create the directories we will use
* Download dataset form kaggle using their API
* Unzip the files we need

Build Features
* Load the unzip data files
* Transpose the data and fill nans with rolling mean or zeroes
* Save the processed data

Train Model
* For the two training datasets
    * Initialize the model
    * Select multistep forecasting range
    * Train the model on each time Series
    * Predict the future web Traffic
* Save the prediction to a text file

Build Submission
* Read in the predictions
* Read in the page ids
* Make a list of datetime
* Combine the predictions with page ids and datetimes
* Melt the dataframe
* Convert the ids and datetimes to hash using the keys
* Save the submission data

Submit data
* Using the kaggle API, submit the submission file

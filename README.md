# Airbnb-London-Rental-Price-Prediction-Project
Will generate Airbnb rental price (per day) prediction for London area if the combined features are not yet offered in the market. Dataset was downloaded from https://insideairbnb.com/get-the-data/.
Note: clean_amenities() in airbnb_london_modelcreate.py will take longer to process since the feature 'amenities' will have to break apart to its substrings which in turn will become new columns (top 50). 
Feel free to edit the values in 'X_test_london_final.csv' to test the price prediction. 
The csv file 'X_test_london_final.csv' will be created in model creation process, therefore, it is imperative to run first the airbnb_london_modelcreate.py Python file.

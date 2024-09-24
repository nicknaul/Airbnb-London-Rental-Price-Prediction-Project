import warnings
import pandas as pd
warnings.simplefilter(action="ignore", category=FutureWarning)
import pickle




def make_prediction():
    df_london_predict = pd.read_csv("X_test_london_final.csv", encoding='latin-1')
    df_london_predict=df_london_predict.drop(columns='Unnamed: 0')
    with open("model-london-airbnb.pkl", "rb") as f:
        model = pickle.load(f)
    predict_price = pd.Series(model.predict(df_london_predict))

    print(predict_price.head(20))


make_prediction()

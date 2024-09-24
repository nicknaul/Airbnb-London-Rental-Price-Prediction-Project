import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)



def wrangle(filepath):

    df = pd.read_csv(filepath,encoding='latin-1')

    df=df[["property_type","neighbourhood_cleansed", "price","latitude","longitude","bedrooms","bathrooms_text","room_type",'amenities']]
    df=df.dropna()
    df["price"]=df["price"].str.replace("$","",regex=False).str.replace(",","",regex=False).astype(float)
    mask_price_high = df["price"] < 1_001
    mask_price_low = df["price"] > 50
    mask_bedrooms =df["bedrooms"] < 6

    df = df[mask_price_high & mask_price_low]
    df=df[df["bathrooms_text"].isin(df["bathrooms_text"].value_counts().head(7).index)]

    low, high = df["price"].quantile([0.1, 0.9])
    mask_price = df["price"].between(low, high)
    df = df[mask_price]
    df = clean_amenities(df)
    return df


def clean_amenities(df):

    df = df.reset_index().drop(columns='index')

    #split 'amenities' description into its individaul substrings
    df_split = df['amenities'].str.rsplit(', ', n=-1, expand=True)

    #remove bracket and double quotation marks
    for i in range(df_split.shape[1]):
        df_split[i] = df_split[i].str.replace("[\"", "")
        df_split[i] = df_split[i].str.replace("\"", "")
        df_split[i] = df_split[i].str.replace("]\"", "")
        df_split[i] = df_split[i].str.replace("]", "")
        df_split[i] = df_split[i].str.replace("[", "")

    #merge all coloumns in df_split by appending each column to column 1
    c=pd.DataFrame()
    for i in range(df_split.shape[1]):
        c=pd.concat([c,df_split[i].to_frame(name='desc')])

    #get the unique values of each substring in 'amenities'
    df_amm=c.value_counts().head(50)
    df_amm=df_amm.reset_index()

    #create new columns in df for the selected substrings
    t=0
    for t in range(len(df_amm)):
        df[df_amm['desc'].iloc[t]]=0

    #assigned values for new column after verifying the presence of the substring in 'amenities'
    v=0
    u=0
    for v in range(len(df)):
            for u in range(len(df_amm)):
                if df_amm['desc'].iloc[u] in df['amenities'].iloc[v]:
                    df.loc[v,df_amm['desc'].iloc[u]]=1
                else:
                    df.loc[v,df_amm['desc'].iloc[u]]=0
    # drop 'amenities' feature
    df.drop(columns="amenities", inplace=True)
    return df




def model_():
    df_london = wrangle("listings.csv")
    target = "price"
    X = df_london.drop([target], axis=1)
    y = df_london[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=20
    )
    y_mean = y_train.mean().round(2)
    y_predict_baseline = [y_mean] * len(y_train)
    ohe = OneHotEncoder(use_cat_names=True)
    ohe.fit(X_train)
    XT_train = ohe.transform(X_train)

    model = make_pipeline(
        OneHotEncoder(use_cat_names=True),
        SimpleImputer(),
        Ridge()
    )

    model.fit(X_train, y_train)
    print("Mean rental price:", y_mean)
    print("Baseline MAE:", mean_absolute_error(y_train, y_predict_baseline).round(2))
    y_pred_training = model.predict(X_train)
    print("Training MAE:", mean_absolute_error(y_train, y_pred_training))
    y_pred_test = pd.Series(model.predict(X_test))
    print(y_pred_test.head())

    with open("model-london-airbnb.pkl", "wb") as f:
       pickle.dump(model, f)
    X_test.to_csv("X_test_london_final.csv")
    
    return print("Model Created")

model_()

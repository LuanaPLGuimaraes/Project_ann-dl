import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET = "Will_Buy_EV"
DATA_PATH = "../../../../data/train.csv"


def load_split(test_size: float = 0.2):
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

if __name__ == "__main__":
    train_df, test_df = load_split()
    print("treino:", train_df.shape)
    print("teste:", test_df.shape)
    print(train_df[TARGET].value_counts(normalize=True))
    print(train_df.dtypes)
    print()
    print(train_df.nunique())
    for col in ["Gender", "City_Type", "Current_Car_Type"]:
        print(col, train_df[col].unique())

# data analysis and types -> nº of uniques
#id                               int64 -> 534932
#Age                              int64 -> 45 
#Annual_Income_USD              float64 -> 12406
#Daily_Commute_km               float64 -> 794
#Number_of_Cars_Owned             int64 -> 4
#Charging_Stations_Near_Home      int64 -> 15
#Charging_Stations_Near_Work      int64 -> 20
#Environmental_Concern_Level    float64 -> 5
#Gender                             str -> 3
#City_Type                          str -> 3
#Current_Car_Type                   str -> 4
#Home_Charging_Possible             str -> 2
#Subsidy_Available                  str -> 2
#Range_Anxiety_Level                str -> 3
#Will_Buy_EV                        str -> 2
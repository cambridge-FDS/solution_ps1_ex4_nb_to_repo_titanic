import numpy as np
import pandas as pd


def missing_data_table_analysis(data):
    """
    Create a table containing total number and percent of missing values for each column.
    """
    total = data.isnull().sum()
    percent = data.isnull().sum() / data.isnull().count() * 100
    tt = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt["Types"] = types
    df_missing = np.transpose(tt)

    return df_missing


def most_freq_table(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ["Total"]
    items = []
    vals = []
    for col in data.columns:
        try:
            itm = data[col].value_counts().index[0]
            val = data[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)
            vals.append(0)
            continue
    tt["Most frequent item"] = items
    tt["Frequence"] = vals
    tt["Percent from total"] = np.round(vals / total * 100, 3)
    return np.transpose(tt)


def unique_values_table(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ["Total"]
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt["Uniques"] = uniques
    return np.transpose(tt)

"""
Reader for Fama-French Data
"""
import pandas as pd

FACTORS = "https://raw.githubusercontent.com/gusamarante/bayesfm/refs/heads/main/sample_data/F-F_Research_Data_5_Factors_2x3.csv"
PORTFOLIOS25 = "https://raw.githubusercontent.com/gusamarante/bayesfm/refs/heads/main/sample_data/25_Portfolios_5x5.CSV"


def get_ff5f():
    """
    Reads and process the CSV that has the Fama-French 5 factors
    """
    ff5f = pd.read_csv(FACTORS,
                       skiprows=2,
                       nrows=729,  # Update this when CSV is updated
                       index_col=0)
    ff5f.index = pd.to_datetime(ff5f.index, format="%Y%m")
    ff5f = ff5f.resample('M').last()
    ff5f = ff5f.drop('RF', axis=1)
    return ff5f


def get_ffrf():
    """
    Grabs the Risk free rate from Fama-French
    """
    rf = pd.read_csv(FACTORS,
                     skiprows=2,
                     nrows=729,  # Update this when CSV is updated
                     index_col=0)
    rf.index = pd.to_datetime(rf.index, format="%Y%m")
    rf = rf.resample('M').last()
    rf = rf['RF']
    return rf


def get_ff25p():
    """
    Reads and process the CSV that has the Fama-French 25 portfolios
    double-sorted on size and value.
    """
    ff25 = pd.read_csv(PORTFOLIOS25,
                       skiprows=15,
                       nrows=1173,  # Update this when CSV is updated
                       index_col=0)
    ff25.index = pd.to_datetime(ff25.index, format="%Y%m")
    ff25 = ff25.resample('M').last()

    size = list(range(1, 6))
    value = list(range(1, 6))

    new_cols = pd.MultiIndex.from_product([size, value],
                                          names=['size', 'value'])
    ff25.columns = new_cols
    return ff25

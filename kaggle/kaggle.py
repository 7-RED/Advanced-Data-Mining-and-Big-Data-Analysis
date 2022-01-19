from scipy import stats
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns
import ipywidgets as widgets
from learntools.time_series.style import *  # plot style settings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

from datetime import date
import holidays
import calendar
import dateutil.easter as easter

from collections import defaultdict
le = defaultdict(LabelEncoder)

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(12, 8))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
%config InlineBackend.figure_format = 'retina'


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import gc
import os
import math
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
PRODUCTION = True # True: For submission run. False: Fast trial run

# Hyperparameters
FOLDS = 15 if PRODUCTION else 3   # Only 5 or 10.
EPOCHS = 300        # Does not matter with Early stopping. Deep network should not take too much epochs to learn
BATCH_SIZE = 2048   # large enough to fit RAM. If unstable, tuned downward. 4096 2048
ACTIVATION = 'swish' # swish mish relu selu ;swish overfit more cause of narrow global minimun
KERNEL_INIT = "glorot_normal" # Minimal impact, but give your init the right foot forward glorot_uniform lecun_normal
LEARNING_RATE = 0.000965713 # Not used. Optimal lr is about half the maximum lr 
LR_FACTOR = 0.5   # LEARNING_RATE * LR_FACTOR = New Learning rate on ReduceLROnPlateau. lower down when the LR oscillate
MIN_DELTA = 0.0000001 # Default 0.0001 0.0000001
RLRP_PATIENCE = 5 # Learning Rate reduction on ReduceLROnPlateau
ES_PATIENCE = 16  # Early stopping
DROPOUT = 0.05     # Act like L1 L2 regulator. lower your learning rate in order to overcome the "boost" that the dropout probability gives to the learning rate.
HIDDEN_LAYERS = [320, 288, 64, 32]

OPTIMIZER = 'adam' # adam adamax nadam
LOSS ='sparse_categorical_crossentropy' # sparse_categorical_crossentropy does not require onehot encoding on labels. categorical_crossentropy
METRICS ='accuracy'  # acc accuracy categorical_accuracy sparse_categorical_accuracy
ACC_VAL_METRICS = 'val_accuracy' # 'val_acc' val_accuracy val_sparse_categorical_accuracy
ACC_METRICS = 'accuracy' # acc accuracy 'sparse_categorical_accuracy'

# The dataset is too huge for trial. Sampling it for speed run!
SAMPLE = 2262087 if PRODUCTION else 11426   # True for FULL run. Max Sample size per category. For quick test: y counts [1468136, 2262087, 195712, 377, 1, 11426, 62261]  # 4000000 total rows
VALIDATION_SPLIT = 0.15 # Only used to min dataset for quick test
MAX_TRIAL = 3           # speed trial any% Not used here
MI_THRESHOLD = 0.001    # Mutual Information threshold value to drop.

RANDOM_STATE = 42
VERBOSE = 0

# Admin
ID = "row_id"            # Id id x X index
INPUT = "../input/tabular-playground-series-jan-2022"
TPU = False           # True: use TPU.
BEST_OR_FOLD = False # True: use Best model, False: use KFOLD softvote
FEATURE_ENGINEERING = True
PSEUDO_LABEL = False # PSEUDO are not ground true and will not help long term, only used for final push
BLEND = False
PSEUDO_DIR = "../input/lolfaker/faker401769.csv"
PSEUDO_DIR2 = "../input/lolfaker/faker401769.csv"

# time series data common new feature  
DATE = "date"
YEAR = "year"
MONTH = "month"
WEEK = "week"
DAY = "day"
DAYOFYEAR = "dayofyear"
DAYOFMONTH = "dayofMonth"
DAYOFWEEK = "dayofweek"
WEEKDAY = "weekday"

assert BATCH_SIZE % 2 == 0, \
    "BATCH_SIZE must be even number."
def smape_loss(y_true, y_pred):
    """
    SMAPE Loss
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute error is returned
        for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.
        SMAPE output is non-negative floating point. The best value is 0.0.

    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)
def better_than_median(inputs, axis):
    """Compute the mean of the predictions if there are no outliers,
    or the median if there are outliers.

    Parameter: inputs = ndarray of shape (n_samples, n_folds)"""
    spread = inputs.max(axis=axis) - inputs.min(axis=axis) 
    spread_lim = 0.45
    print(f"Inliers:  {(spread < spread_lim).sum():7} -> compute mean")
    print(f"Outliers: {(spread >= spread_lim).sum():7} -> compute median")
    print(f"Total:    {len(inputs):7}")
    return np.where(spread < spread_lim,
                    np.mean(inputs, axis=axis),
                    np.median(inputs, axis=axis))
def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax
def impute(df):
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    return df
def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.DataFrame(features, index=index)
def get_basic_ts_features(df):
    
    gdp_df = pd.read_csv('../input/gdp-20152019-finland-norway-and-sweden/GDP_data_2015_to_2019_Finland_Norway_Sweden.csv')
    gdp_df.set_index('year', inplace=True)
    gdp_exponent = 1.2121103201489674 # see https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model for an explanation
    def get_gdp(row):
        country = 'GDP_' + row.country
        return gdp_df.loc[row.date.year, country]
    
    df['gdp'] = np.log1p(df.apply(get_gdp, axis=1))
    df['season'] = ((df[DATE].dt.month % 12 + 3) // 3).map({1:'DJF', 2: 'MAM', 3:'JJA', 4:'SON'})
#     df[MONTH] = df[MONTH].apply(lambda x: calendar.month_abbr[x])

    df['wd4'] = df[DATE].dt.weekday == 4
#     df['wd5'] = df[DATE].dt.weekday == 5
#     df['wd6'] = df[DATE].dt.weekday >= 6
    df['wd56'] = df[DATE].dt.weekday >= 5
    
#     df.loc[(df.date.dt.year != 2016) & (df.date.dt.month >=3), DAYOFYEAR] += 1 # fix for leap years
    
    # 21 days cyclic for lunar
    dayofyear = df.date.dt.dayofyear
    for k in range(1, 3, 1):
        df[f'sin{k}'] = np.sin(dayofyear / 365 * 2 * math.pi * k)
        df[f'cos{k}'] = np.cos(dayofyear / 365 * 2 * math.pi * k)
        df[f'finland_sin{k}'] = np.where(df['country'] == 'Finland', df[f'sin{k}'], 0)
        df[f'finland_cos{k}'] = np.where(df['country'] == 'Finland', df[f'cos{k}'], 0)
        df[f'norway_sin{k}'] = np.where(df['country'] == 'Norway', df[f'sin{k}'], 0)
        df[f'norway_cos{k}'] = np.where(df['country'] == 'Norway', df[f'cos{k}'], 0)
        df[f'store_sin{k}'] = np.where(df['country'] == 'KaggleMart', df[f'sin{k}'], 0)
        df[f'store_cos{k}'] = np.where(df['country'] == 'KaggleMart', df[f'cos{k}'], 0)
        df[f'mug_sin{k}'] = np.where(df['country'] == 'Kaggle Mug', df[f'sin{k}'], 0)
        df[f'mug_cos{k}'] = np.where(df['country'] == 'Kaggle Mug', df[f'cos{k}'], 0)
#         df[f'sticker_sin{k}'] = np.where(df['country'] == 'Kaggle Sticker', df[f'sin{k}'], 0)
#         df[f'sticker_cos{k}'] = np.where(df['country'] == 'Kaggle Sticker', df[f'cos{k}'], 0)
    
    df[f'semiweekly_sin'] = np.sin(dayofyear / 365 * 2 * math.pi * 14)
    df[f'semiweekly_cos'] = np.cos(dayofyear / 365 * 2 * math.pi * 14)
    df[f'lunar_sin'] = np.sin(dayofyear / 365 * 2 * math.pi * 21)
    df[f'lunar_cos'] = np.cos(dayofyear / 365 * 2 * math.pi * 21)
    df[f'season_sin'] = np.sin(dayofyear / 365 * 2 * math.pi * 91.5)
    df[f'season_cos'] = np.cos(dayofyear / 365 * 2 * math.pi * 91.5)
#     df = pd.concat([df, pd.DataFrame({f'fin{ptr[1]}':
#                                       (df.date == pd.Timestamp(ptr[0])) & (df.country == 'Finland')
#                                       for ptr in holidays.Finland(years = [2015,2016,2017,2018,2019]).items()})], axis=1)
#     df = pd.concat([df, pd.DataFrame({f'nor{ptr[1]}':
#                                       (df.date == pd.Timestamp(ptr[0])) & (df.country == 'Norway')
#                                       for ptr in holidays.Norway(years = [2015,2016,2017,2018,2019]).items()})], axis=1)
#     df = pd.concat([df, pd.DataFrame({f'swe{ptr[1]}':
#                                       (df.date == pd.Timestamp(ptr[0])) & (df.country == 'Sweden')
#                                       for ptr in holidays.Sweden(years = [2015,2016,2017,2018,2019]).items()})], axis=1)

    # End of year
    # Dec
    # End of year
    df = pd.concat([df,
                        pd.DataFrame({f"dec{d}":
                                      (df.date.dt.month == 12) & (df.date.dt.day == d)
                                      for d in range(24, 32)}),
                        pd.DataFrame({f"n-dec{d}":
                                      (df.date.dt.month == 12) & (df.date.dt.day == d) & (df.country == 'Norway')
                                      for d in range(24, 32)}),
                        pd.DataFrame({f"f-jan{d}":
                                      (df.date.dt.month == 1) & (df.date.dt.day == d) & (df.country == 'Finland')
                                      for d in range(1, 14)}),
                        pd.DataFrame({f"jan{d}":
                                      (df.date.dt.month == 1) & (df.date.dt.day == d) & (df.country == 'Norway')
                                      for d in range(1, 10)}),
                        pd.DataFrame({f"s-jan{d}":
                                      (df.date.dt.month == 1) & (df.date.dt.day == d) & (df.country == 'Sweden')
                                      for d in range(1, 15)})
                       ], axis=1)
        
    # May
    df = pd.concat([df,
                        pd.DataFrame({f"may{d}":
                                      (df.date.dt.month == 5) & (df.date.dt.day == d) 
                                      for d in list(range(1, 10))}),
                        pd.DataFrame({f"may{d}":
                                      (df.date.dt.month == 5) & (df.date.dt.day == d) & 
                                      (df.country == 'Norway')
                                      for d in list(range(18, 28))})
                        ], axis=1)
    
    # June and July
    df = pd.concat([df,
                        pd.DataFrame({f"june{d}":
                                      (df.date.dt.month == 6) & (df.date.dt.day == d) & 
                                      (df.country == 'Sweden')
                                      for d in list(range(8, 14))}),
                       ], axis=1)
    
    #Swedish Rock Concert
    #Jun 3, 2015 – Jun 6, 2015
    #Jun 8, 2016 – Jun 11, 2016
    #Jun 7, 2017 – Jun 10, 2017
    #Jun 6, 2018 – Jun 10, 2018
    #Jun 5, 2019 – Jun 8, 2019
    swed_rock_fest  = df.date.dt.year.map({2015: pd.Timestamp(('2015-06-6')),
                                         2016: pd.Timestamp(('2016-06-11')),
                                         2017: pd.Timestamp(('2017-06-10')),
                                         2018: pd.Timestamp(('2018-06-10')),
                                         2019: pd.Timestamp(('2019-06-8'))})
    df = pd.concat([df, pd.DataFrame({f"swed_rock_fest{d}":
                                      (df.date - swed_rock_fest == np.timedelta64(d, "D")) & (df.country == 'Sweden')
                                      for d in list(range(-3, 3))})], axis=1)

    
    # Last Wednesday of June
    wed_june_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-06-24')),
                                         2016: pd.Timestamp(('2016-06-29')),
                                         2017: pd.Timestamp(('2017-06-28')),
                                         2018: pd.Timestamp(('2018-06-27')),
                                         2019: pd.Timestamp(('2019-06-26'))})
    df = pd.concat([df, pd.DataFrame({f"wed_june{d}": 
                                      (df.date - wed_june_date == np.timedelta64(d, "D")) & 
                                      (df.country != 'Norway')
                                      for d in list(range(-4, 6))})], axis=1)
        
    # First Sunday of November
    sun_nov_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-11-1')),
                                         2016: pd.Timestamp(('2016-11-6')),
                                         2017: pd.Timestamp(('2017-11-5')),
                                         2018: pd.Timestamp(('2018-11-4')),
                                         2019: pd.Timestamp(('2019-11-3'))})
    df = pd.concat([df, pd.DataFrame({f"sun_nov{d}":
                                      (df.date - sun_nov_date == np.timedelta64(d, "D")) & (df.country == 'Norway')
                                      for d in list(range(0, 9))})], axis=1)
    # First half of December (Independence Day of Finland, 6th of December)
    df = pd.concat([df, pd.DataFrame({f"dec{d}":
                                      (df.date.dt.month == 12) & (df.date.dt.day == d) & (df.country == 'Finland')
                                      for d in list(range(6, 14))})], axis=1)
    # Easter
    easter_date = df.date.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    df = pd.concat([df, pd.DataFrame({f"easter{d}":
                                      (df.date - easter_date == np.timedelta64(d, "D"))
                                      for d in list(range(-2, 11)) + list(range(40, 48)) + list(range(50, 59))})], axis=1)
    
    return df
for ptr in holidays.Norway(years = [2019], observed=True).items():
    print(ptr)
def feature_engineer(df):
    df = get_basic_ts_features(df)
    return df
    
from pathlib import Path


def load_data():
    # Read data
    data_dir = Path(INPUT)
    df_train = pd.read_csv(data_dir / "train.csv", parse_dates=[DATE],
                    usecols=['date', 'country', 'store', 'product', 'num_sold'],
                    dtype={
                        'country': 'category',
                        'store': 'category',
                        'product': 'category',
                        'num_sold': 'float32',
                    },
                    infer_datetime_format=True,)
    df_test = pd.read_csv(data_dir / "test.csv", index_col=ID, parse_dates=[DATE])
    column_y = df_train.columns.difference(
        df_test.columns)[0]  # column_y target_col label_col
    df_train[DATE] = pd.to_datetime(df_train[DATE])
    df_test[DATE] = pd.to_datetime(df_test[DATE])
    return df_train, df_test, column_y
def process_data(df_train, df_test):
    # Preprocessing
#     df_train = impute(df_train)
#     df_test = impute(df_test)
    
    if FEATURE_ENGINEERING:
        df_train = feature_engineer(df_train)
        df_test = feature_engineer(df_test)
    
#     df_train = reduce_mem_usage(df_train)
#     df_test = reduce_mem_usage(df_test)

    return df_train, df_test
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
        
        
%%time
train_df, test_df, column_y = load_data()
%%time
train_df, test_df = process_data(train_df, test_df)
        
        
train_data = train_df.copy()
train_data[DATE] = train_df.date.dt.to_period('D')
test_data = test_df.copy()
test_data[DATE] = test_df.date.dt.to_period('D')

df_pseudolabels = pd.read_csv(PSEUDO_DIR, index_col=ID)
df_pseudolabels[DATE] = pd.to_datetime(test_df[DATE])
df_pseudolabels.to_csv("pseudo_labels_v0.csv", index=True)
# if PSEUDO_LABEL:
    # df_pseudolabels = df_pseudolabels.set_index([DATE]).sort_index()
test_data[column_y] = df_pseudolabels[column_y].astype(np.float32)
train_data = pd.concat([train_data, test_data], axis=0)
train_df = pd.concat([train_df, test_data], axis=0)
X = train_data.set_index([DATE]).sort_index()
X_test = test_data.set_index([DATE]).sort_index()
train_data = train_data.set_index(['date', 'country', 'store', 'product']).sort_index()
frames = [kaggle_sales_2015, kaggle_sales_2016, kaggle_sales_2017, kaggle_sales_2018]
kaggle_sales = pd.concat(frames)
missing_val = X.isnull().sum()
print(missing_val[missing_val > 0])
train_data.groupby(column_y).apply(lambda s: s.sample(min(len(s), 5)))
fig_dims = (30,30)
ax = kaggle_sales.num_sold.plot(title='Sales Trends', figsize=fig_dims)
_ = ax.set(ylabel="Numbers sold")
def show_me(data) :
    fig_dims = (20,10)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.set_theme(style="whitegrid")
    dates = pd.date_range("1 1 2015", periods=365, freq="D")
    dates = pd.date_range(start='1/1/2015', end='31/12/2016',  freq="D")
    data.index = dates
    sns.lineplot(data=data, palette="tab10", linewidth=1)
plot_periodogram(X[column_y]);
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax
import matplotlib.dates as mdates

# Plot all num_sold_true and num_sold_pred (five years) for one country-store-product combination
def plot_five_years_combination(engineer, country='Norway', store='KaggleMart', product='Kaggle Hat', period_start='2015-01-01', period_end='2019-12-31'):
    locator = mdates.AutoDateLocator(minticks=60)
    dtFmt = mdates.ConciseDateFormatter(locator)
    
    demo_df = pd.DataFrame({'row_id': 0,
                            'date': pd.date_range(period_start, period_end, freq='D'),
                            'country': country,
                            'store': store,
                            'product': product})
    demo_df.set_index('date', inplace=True, drop=False)
    demo_df = engineer(demo_df)
    demo_df['num_sold'] = model.predict(demo_df[features])
    train_subset = X[(X.country == country) & (X.store == store) & (X['product'] == product)].copy()
    train_subset = train_subset.loc[period_start:period_end]
    fig, ax = plt.subplots(figsize=(32, 8))
    plt.plot(demo_df[DATE], demo_df.num_sold, label='prediction', alpha=0.5, color='blue')
    plt.plot(train_subset.index, train_subset.num_sold, label='true', alpha=0.3, color='red', linestyle='--')
    plt.scatter(train_subset.index, train_subset.num_sold, label='true', alpha=0.3, color='red', s=2)
    plt.grid(True)
    plt.grid(which='major',axis ='y', linestyle=':', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax.xaxis.set_major_formatter(dtFmt) # apply the format to the desired axis
    ax.xaxis.set_major_locator(locator)
    
    plt.legend()
    plt.title(f'{country} {store} {product} Predictions and true from {period_start} to {period_end}')
    plt.tight_layout()
    plt.show()
    return demo_df['num_sold']
def find_min_SMAPE(y_true, y_predict):
    loss_correction = 1
    scores = []
    # float step
    for WEIGHT in np.arange(0.97, 1.02, 0.0001):
        y_hat = y_predict.copy()
        y_hat *= WEIGHT
        scores.append(np.array([WEIGHT, np.mean(smape_loss(y_true, y_hat))]))
        
    scores = np.vstack(scores)
    min_SMAPE = np.min(scores[:,1])
    print(f'min SMAPE {min_SMAPE:.5f}')
    for x in scores:
        if x[1] == min_SMAPE:
            loss_correction = x[0]
            print(f'loss_correction: {x[0]:.5f}')
            
    plt.figure(figsize=(5, 3))
    plt.plot(scores[:,0],scores[:,1])
    plt.scatter([loss_correction], [min_SMAPE], color='g')
    plt.ylabel(f'SMAPE')
    plt.xlabel(f'loss_correction: {loss_correction:.5f}')
    plt.legend()
    plt.title(f'min SMAPE:{min_SMAPE:.5f} scaling')
    plt.show()
    
    return loss_correction
def plot_true_vs_prediction(df_true, df_hat):
    plt.figure(figsize=(20, 13))
    plt.scatter(np.arange(len(df_hat)), np.log1p(df_hat), label='prediction', alpha=0.5, color='blue', s=3) #np.arange(len(df_hat))
    plt.scatter(np.arange(len(df_true)), np.log1p(df_true), label='Pseudo/true', alpha=0.5, color='red', s=7) #np.arange(len(df_true))
    plt.legend()
    plt.title(f'Predictions VS Pseudo-label {column_y} (LOG)') #{df_true.index[0]} - {df_true.index[-1]}
    plt.show()
def plot_residuals(y_residuals):
    plt.figure(figsize=(13, 3))
    plt.scatter(np.arange(len(y_residuals)), y_residuals, label='residuals', alpha=0.1, color='blue', s=5)
    plt.legend()
    plt.title(f'Linear Model residuals {column_y} (LOG)') #{df_true.index[0]} - {df_true.index[-1]}
    plt.tight_layout()
    plt.show()
def plot_oof(y_true, y_predict):
    # Plot y_true vs. y_pred
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_predict, s=3, color='r', alpha=0.5)
#     plt.scatter(np.log1p(y_true), np.log1p(y_predict), s=1, color='g', alpha=0.3)
    plt.plot([plt.xlim()[0], plt.xlim()[1]], [plt.xlim()[0], plt.xlim()[1]], '--', color='k')
    plt.gca().set_aspect('equal')
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    plt.title('OOF Predictions')
    plt.show()
def evaluate_SMAPE(y_va, y_va_pred):
    loss_correction = 1
    # Evaluation: Execution time and SMAPE
    smape_before_correction = np.mean(smape_loss(y_va, y_va_pred))
    smape = np.mean(smape_loss(y_va, y_va_pred))
    loss_correction = find_min_SMAPE(y_va, y_va_pred)
    y_va_pred *= loss_correction
    print(f"SMAPE (before correction: {smape_before_correction:.5f})")
    print(f'Min SMAPE: {np.mean(smape_loss(y_va, y_va_pred))}')
    return loss_correction
def evaluate(model, X, y, cv):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
    )
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    print(
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
    )
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn import set_config
set_config(display='diagram') 

# Model 1 (trend)
from pyearth import Earth
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, HuberRegressor, RidgeCV, TheilSenRegressor, RANSACRegressor

# Model 2
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class BoostedHybrid(BaseEstimator, RegressorMixin):
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None  # store column names from fit method
    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        # Train model_1
        self.model_1.fit(X, y)

        # Make predictions
        y_fit = self.model_1.predict(X)
        # Compute residuals
        y_resid = y - y_fit

        # Train model_2 on residuals , eval_set=[(X_1_valid, y_valid_resid)]
        self.model_2.fit(X, y_resid)
        # Model2 prediction
        y_fit2 = self.model_2.predict(X)
        # Compute noise
        y_resid2 = y_resid - y_fit2
        
        # Save data for question checking
        self.y = y
        self.y_fit = y_fit
        self.y_resid = y_resid
        self.y_fit2 = y_fit2
        self.y_resid2 = y_resid2

        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        # Predict with model_1
        y_predict = self.model_1.predict(X)
        # Add model_2 predictions to model_1 predictions
        y_predict += self.model_2.predict(X)

        return y_predict
def model_fit_eval(hybrid_model, X_train, y_train, X_valid, y_valid, loss_correction):
    test_pred_list = []
    # Boosted Hybrid
    hybrid_model.fit(X_train, y_train) #, X_valid, y_valid
    y_va_pred = hybrid_model.predict(X_valid)
    
    loss_correction = 1
    ###### Preprocess the validation data
    y_va = y_valid.copy()
    # Inference for validation
    y_va_pred = hybrid_model.predict(X_valid)
    loss_correction = evaluate_SMAPE(y_va, y_va_pred)
    
    ###### Visualize and evual
    plot_oof(y_va, y_va_pred)
    plot_true_vs_prediction(y_va, y_va_pred)
#     plot_residuals(hybrid_model.y_resid)
#     plot_residuals(hybrid_model.y_resid2)
    # plot_residuals(model.y_resid3)
    
    ###### Validate against 2019 PSEU #######
    loss_correction = 1
    ###### Preprocess the validation data
    y_va = df_pseudolabels[column_y].values.reshape(-1, 1)
    
    # Inference test 2019 for validation
    y_va_pred = hybrid_model.predict(X_test[features]) #TODO
    
    # Evaluation: Execution time and SMAPE
    smape_before_correction = np.mean(smape_loss(y_va, y_va_pred.reshape(-1, 1)))
    smape = np.mean(smape_loss(y_va, y_va_pred.reshape(-1, 1)))
    print(f'***********Test Data*****************')
    loss_correction = find_min_SMAPE(y_va, y_va_pred.reshape(-1, 1))
#     y_va_pred *= loss_correction
    
    ### Mean test prediction ###
    test_pred_list.append(y_va_pred)

    print(f'SMAPE (before correction: {smape_before_correction:.5f})')
    print(f'Min SMAPE: {np.mean(smape_loss(y_va, y_va_pred.reshape(-1, 1)*loss_correction))}')
    
    return hybrid_model, test_pred_list, loss_correction
y = X.loc[:, column_y]
X_2 = X.drop(column_y, axis=1)
features = X_2.columns
if PSEUDO_LABEL:
    TRAIN_END_DATE = "2019-12-31"
    VALID_START_DATE = "2015-01-01"
    VALID_END_DATE = "2018-12-31"
else:
    if PRODUCTION:
        TRAIN_END_DATE = "2018-12-31"
    else:
        TRAIN_END_DATE = "2017-12-31"
    VALID_START_DATE = "2018-01-01"
    VALID_END_DATE = "2018-12-31"

y_train, y_valid = y[:TRAIN_END_DATE], y[VALID_START_DATE:VALID_END_DATE]
X2_train, X2_valid = X_2.loc[:TRAIN_END_DATE], X_2.loc[VALID_START_DATE:VALID_END_DATE]
%%time
gc.collect()
LOSS_CORRECTION = 1
estimator_stack = []

param1 = {  'loss_function' : 'MultiRMSE',
            'eval_metric': 'MultiRMSE',
            'n_estimators': 1000,
            'od_type' : 'Iter',
            'od_wait' : 20,
            'random_state': RANDOM_STATE,
            'verbose': VERBOSE
         }

if PRODUCTION:
    # Linear estimator. Try different combinations of the algorithms above KNeighborsRegressor fit_intercept=False , fit_intercept=False
    models_1 = [
                ElasticNet(fit_intercept=False, random_state=RANDOM_STATE),
                Ridge(fit_intercept=False, random_state=RANDOM_STATE),
                LinearRegression(fit_intercept=False),
                MLPRegressor(   hidden_layer_sizes=(200, 100),
                                learning_rate_init=0.01,
                                early_stopping=True,
                                max_iter=EPOCHS,
                                random_state=RANDOM_STATE,
                                ),
               ]
    # Residue estimator
    models_2 = [
                XGBRegressor(objective='reg:pseudohubererror', tree_method='hist', n_estimators=1000),
                lgb.LGBMRegressor(objective='regression', n_estimators=1000, random_state=RANDOM_STATE),
                CatBoostRegressor(**param1),
               ]
else:
    # Linear estimator. Try different combinations of the algorithms above KNeighborsRegressor fit_intercept=False , fit_intercept=False
    models_1 = [
                Ridge(fit_intercept=False, random_state=RANDOM_STATE),
                LinearRegression(fit_intercept=False),
                MLPRegressor(   hidden_layer_sizes=(200, 100),
                                learning_rate_init=0.01,
                                learning_rate='adaptive',
                                early_stopping=True,
                                max_iter=EPOCHS,
                                random_state=RANDOM_STATE,
                                ),
#                 TheilSenRegressor(fit_intercept=False, random_state=RANDOM_STATE),
               ]
    # Residue estimator
    models_2 = [
                lgb.LGBMRegressor(objective='regression', n_estimators=1000, random_state=RANDOM_STATE),
                CatBoostRegressor(**param1),
                XGBRegressor(objective='reg:pseudohubererror', tree_method='hist', n_estimators=1000),
               ]

for model_1 in models_1:
    for model_2 in models_2:
        model1_name = type(model_1).__name__
        model2_name = type(model_2).__name__
        hybrid_model = BoostedHybrid(
                model_1 = model_1,
                model_2 = model_2
                        )
        print(f'******************Stacking {model1_name:>15} with {model2_name:<18}*************************')
        estimator_stack.append((f'model_{model1_name}_{model2_name}', hybrid_model))
one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
%%time
# tscv = TimeSeriesSplit(n_splits=FOLDS)
# X , cv=tscv
stacking_regressor = make_pipeline(
    ColumnTransformer(
        transformers=[
            ("categorical", one_hot_encoder, make_column_selector(dtype_include=object)),
            ("numeric", MinMaxScaler(), make_column_selector(dtype_include=np.number)),
        ],
        remainder='passthrough',
    ),
    StackingRegressor(estimators=estimator_stack, final_estimator=RidgeCV(), cv=FOLDS, n_jobs=-1, verbose=VERBOSE),
)
# X y
model = TransformedTargetRegressor(
    regressor=stacking_regressor, func=np.log1p, inverse_func=np.expm1
)

model, test_pred_list, LOSS_CORRECTION = model_fit_eval(model, X2_train, y_train, X2_valid[features], y_valid, LOSS_CORRECTION) 
model
for ptr in holidays.Norway(years = [2018], observed=True).items():
    print(ptr)
for ptr in holidays.Finland(years = [2018], observed=True).items():
    print(ptr)
for country in np.unique(train_df['country']):
    for product in np.unique(train_df['product']):
        for store in np.unique(train_df['store']):
            y_fit = plot_five_years_combination(feature_engineer, country=country, product=product, store=store,period_start='2017-09-01', period_end='2017-09-30')
            break
        break
for country in np.unique(train_df['country']):
    for product in np.unique(train_df['product']):
        for store in np.unique(train_df['store']):
            y_fit = plot_five_years_combination(feature_engineer, country=country, product=product, store=store)
y_pred = sum(test_pred_list) / len(test_pred_list) #model.predict(X_test[features])
%%time
LOSS_CORRECTION = 1

###### Preprocess the validation data
y_va = df_pseudolabels[column_y].values.reshape(-1, 1)

# Inference for validation
y_va_pred = y_pred.copy().reshape(-1, 1) #model.predict(X_test[features])

# Evaluation: Execution time and SMAPE
smape_before_correction = np.mean(smape_loss(y_va, y_va_pred))
smape = np.mean(smape_loss(y_va, y_va_pred))
LOSS_CORRECTION = find_min_SMAPE(y_va, y_va_pred)
y_va_pred *= LOSS_CORRECTION

print(f" SMAPE: {smape:.5f} (before correction: {smape_before_correction:.5f})")
print(np.mean(smapplot_oof(y_va, y_va_pred)
plot_true_vs_prediction(y_va, y_va_pred)
plot_residuals(model.regressor_[1].estimators_[0].y_resid)
plot_residuals(model.regressor_[1].estimators_[0].y_resid2)e_loss(y_va, y_va_pred)))
from math import ceil, floor, sqrt
# from https://www.kaggle.com/fergusfindley/ensembling-and-rounding-techniques-comparison
def geometric_round(arr):
    result_array = arr
    result_array = np.where(result_array < np.sqrt(np.floor(arr)*np.ceil(arr)), np.floor(arr), result_array)
    result_array = np.where(result_array >= np.sqrt(np.floor(arr)*np.ceil(arr)), np.ceil(arr), result_array)

    return result_array
sub = pd.read_csv('../input/tabular-playground-series-jan-2022/sample_submission.csv')
# Inference for test
test_prediction_list = []
test_prediction_list.append(y_pred) # * LOSS_CORRECTION)
if BLEND:
    test_prediction_list.append(df_pseudolabels[column_y].values) #blender 1
#     df_pseudolabels1 = pd.read_csv(PSEUDO_DIR2, index_col=ID)    
#     test_prediction_list.append(df_pseudolabels1[column_y].values) #blender 2
test_prediction_list = np.median(test_prediction_list, axis=0) # median is better https://www.kaggle.com/saraswatitiwari/tabular-playground-series-22

if len(test_prediction_list) > 0:
    # Create the submission file
    submission = pd.DataFrame(data=np.zeros((sub.shape[0],2)),index = sub.index.tolist(),columns=[ID,column_y])
    submission[ID] = sub[ID]
    submission[column_y] = test_prediction_list
    submission[column_y] = geometric_round(submission[column_y]).astype(int) #https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/299162
    submission.to_csv('submission.csv', index=False)

    # Plot the distribution of the test predictions
    plt.figure(figsize=(16,3))
    plt.hist(train_df[column_y], bins=np.linspace(0, 3000, 201),
             density=True, label='Training')
    plt.hist(submission[column_y], bins=np.linspace(0, 3000, 201),
             density=True, rwidth=0.5, label='Test predictions')
    plt.xlabel(column_y)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
display(submission.head(30))
display(submission.tail(30))

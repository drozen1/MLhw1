import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.model_selection import train_test_split
import warnings
import scipy.stats as stats

warnings.filterwarnings('ignore')


def remove_outliers(data, columns):
    all_indexes=[]
    x = np.zeros(data.shape[0], dtype=bool)
    for i in (range(len(columns))):
        if( isinstance(data[columns[i]][0], str) ):
            pass
        else:
            if(columns[i] != "DateOfPCRTest"):
                npdata = np.asarray(data[columns[i]])
                b = stats.zscore(npdata)
                x = ((np.absolute(b) >= 3) | x)
    afterDrop = data[(x == False)]
    return afterDrop


# function for replacing missing data in train and for turning everything into numeric data
def replace_to_mean(data):
    # for future use:
    intTypeList = ['AgeGroup', 'NrCousins' 'StepsPerYear']
    tp = data.dtype
    if data.name in intTypeList:
        new_data = data.fillna(round(data.mean()))
    elif tp == "float64":
        new_data = data.fillna(data.mean())
    else:
        new_data = data.fillna(data.mode()[0])
        # numeric_temp = range(len(temp))

        # for i in range(len(data)):
        #     if data[i].isnull() == False:
        #         new_data[i] = numeric_temp[temp == data[i]]
        #     new_data = new_data.fillna(new_data.mean())
    return new_data


# function for creating two features of X and Y location coordinates instead of string CurrentLocation
def split_in_XY(data):
    dataX = []
    dataY = []
    for i in data:
        if i != i:
            dataX.insert(len(dataX), np.nan)
            dataY.insert(len(dataX), np.nan)
        else:
            dataX.insert(len(dataX), float(i.split("Decimal")[1][2:-4]))
            dataY.insert(len(dataY), float(i.split("Decimal")[2][2:-4]))
    return pd.Series(dataX), pd.Series(dataY)


# function for checking amount of NANs in columns
def NAN_checker(data):
    check_null = data.isnull()
    columns = list(check_null)
    percentNAN = [0] * len(columns)
    for i in (range(len(columns))):
        print(columns[i])
        check_null_col = check_null[columns[i]]
        percentNAN[i] = 100 * check_null_col.value_counts(normalize=True)
        print(percentNAN[i])


if __name__ == '__main__':

    # Part 1.1 - importing the data
    filename = 'virus_hw1.csv'
    dataset = pd.read_csv(filename)
    datasetCopy = dataset.copy()

    # Part 1.3: changing to the correct type
    # Before changing, we'd like to see the data in graphs for quick and efficient decisions

    # TODO write all types in the table and explanations why add split function
    print(dataset.info())  # to properly evaluate the types we'd like to see the data information
    dataset.Address = dataset.Address.astype('string')
    # dataset.AgeGroup = dataset.AgeGroup.astype('category')
    dataset.BloodType = dataset.BloodType.astype('category')
    dataset.DateOfPCRTest = dataset.DateOfPCRTest.astype('datetime64')
    dataset.Job = dataset.Job.astype('string')
    # dataset.NrCousins = dataset.NrCousins.astype('category')
    dataset.Sex = dataset.Sex.astype('category')
    dataset.Virus = dataset.Virus.astype('category')
    dataset.Risk = dataset.Risk.astype('category')
    dataset.SpreadLevel = dataset.SpreadLevel.astype('category')

    # Part 2.4: splitting. As splitting is into 2 sets, we split into 3 in two steps: 60/40 and then 50/50 for the 40
    train, test_temp = train_test_split(dataset, test_size=0.4, random_state=14)
    test, validate = train_test_split(test_temp, test_size=0.5, random_state=14)

    # Part 2.5: already done in Part 1.3

    # Part 2.6: discussion is in the attached file
    train.X, train.Y = split_in_XY(train.CurrentLocation)
    train.Virus = train.Virus.astype('object')
    train.Virus[(train.Virus == 'covid')] = '1'
    train.Virus[(train.Virus != '1')] = '0'
    train.Virus = train.Virus.astype('category')
    # We are keeping the following line of code in case diffrentiating between not covid and other illnesses is important:
    # train.Virus[(train.Virus != 'covid') & (train.Virus != 'not_detected')] = 'other'
    # next line is for evaluating NAN in columns, were used mainly for JOB.
    # NAN_checker(data)
    # removing data as discussed in attached file
    train = train.drop(['ID', 'Address', 'CurrentLocation', 'Job'], axis=1)

    # Part 2.7: discussion is in the attached file
    # TODO remove according to table + show NAN amounts + change to train
    # we'd like to find the amount of missing data in each feature:
    NAN_checker(train)
    train = train.drop(['PCR_11', 'PCR_15'], axis=1)

    # Part 2.8: Discussion in attached file
    # First, we'd like to change the data into numeric data for easier handling and later modelling.
    columns = list(train)

    for i in (range(len(columns))):
        if columns[i] == 'DateOfPCRTest':
            DateOfBase = train.DateOfPCRTest
            minDate = DateOfBase.min()
            avg_date = (minDate + (DateOfBase - minDate).mean())
            train.DateOfPCRTest = train.DateOfPCRTest.fillna(avg_date)

        else:
            train[columns[i]] = replace_to_mean(train[columns[i]])

    # outliers
    train = remove_outliers(train, columns)

    sns.histplot(train.BMI, bins=100, kde=True)
    plt.grid()
    plt.savefig('train_BMI_histogram.png')
    plt.close()

    # All plots required for this assignment were made here:
    # BMI histogram
    sns.histplot(dataset.BMI, bins=100, kde=True)
    plt.grid()
    plt.savefig('BMI_histogram.png')
    plt.close()

    # Blood type bins:
    datasetCopy.Virus[(datasetCopy.Virus != 'covid') & (datasetCopy.Virus != 'not_detected')] = 'other'
    BloodType_plot = pd.crosstab([datasetCopy.BloodType], datasetCopy.Virus)
    BloodType_plot.plot(kind='bar', stacked=True, color=['red', 'blue', 'green'], grid=True)
    plt.grid()
    plt.savefig('Bloodtype_histogram.png')
    plt.close()

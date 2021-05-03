import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeClassifier
import warnings
import scipy.stats as stats
import math
import re

warnings.filterwarnings('ignore')

#function for convert some features to numeric values
def convertToNumerical(train):
    bloodTypes = pd.Series(train.BloodType).unique()
    count = 0
    for i in bloodTypes:
        train.BloodType.replace(to_replace=i, value=float(count), inplace=True)
        count += 1
    train.Sex.replace(to_replace="F", value=float(1), inplace=True)
    train.Sex.replace(to_replace="M", value=float(0), inplace=True)
    viruses = pd.Series(train.Virus).unique()
    count = 0
    for i in viruses:
        train.Virus.replace(to_replace=i, value=float(count), inplace=True)
        count += 1
    risks = pd.Series(train.Risk).unique()
    count = 0
    for i in risks:
        train.Risk.replace(to_replace=i, value=float(count), inplace=True)
        count += 1
    spreadLevels = pd.Series(train.SpreadLevel).unique()
    count = 0
    for i in spreadLevels:
        train.SpreadLevel.replace(to_replace=i, value=float(count), inplace=True)
        count += 1
    return train

#function for remove outliers
def remove_outliers(data, columns):
    x = np.zeros(data.shape[0], dtype=bool)
    for i in (range(len(columns))):
        if isinstance(data[columns[i]][0], str):
            pass
        else:
            npdata = np.asarray(data[columns[i]])
            b = stats.zscore(npdata)
            data[columns[i]] = b  # remove this line for generating graphs
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
        # print(columns[i])
        check_null_col = check_null[columns[i]]
        percentNAN[i] = 100 * check_null_col.value_counts(normalize=True)
        # print(percentNAN[i])

def isNaN(string):
    return string != string


if __name__ == '__main__':

    # Part 1.1 - importing the data
    filename = 'virus_hw1.csv'
    dataset = pd.read_csv(filename)
    datasetCopy = dataset.copy()



    # Part 1.3: changing to the correct type
    # Before changing, we'd like to see the data in graphs for quick and efficient decisions. That's why there is some lines in comment
    #print(dataset.info())  # to properly evaluate the types we'd like to see the data information
    dataset.Address = dataset.Address.astype('string')
    # dataset.AgeGroup = dataset.AgeGroup.astype('category')
    # dataset.BloodType = dataset.BloodType.astype('category')
    dataset.DateOfPCRTest = dataset.DateOfPCRTest.astype('datetime64')
    dataset.Job = dataset.Job.astype('string')
    # dataset.NrCousins = dataset.NrCousins.astype('category')
    # dataset.Sex = dataset.Sex.astype('category')
    #  dataset.Virus = dataset.Virus.astype('category')
    #  dataset.Risk = dataset.Risk.astype('category')
    #  dataset.SpreadLevel = dataset.SpreadLevel.astype('category')

    # Part 2.4: splitting. As splitting is into 2 sets, we split into 3 in two steps: 60/40 and then 50/50 for the 40
    train, test_temp = train_test_split(dataset, test_size=0.4, random_state=14)
    test, validate = train_test_split(test_temp, test_size=0.5, random_state=14)

    # Part 2.5: already done in Part 1.3

    # Part 2.6: discussion is in the attached file

    x_orig, y_orig = split_in_XY(train.CurrentLocation)
    X = x_orig.copy()
    Y = y_orig.copy()
    train['X'] = X
    train['Y'] = Y

    # splitting Self_declaration_of_Illness_Form for future use:
    illness_types = set()
    col_Illness_Form = train.Self_declaration_of_Illness_Form.copy()
    for i in col_Illness_Form:
        if isNaN(i) == False:
            lst = i.split("; ")
            for j in lst:
                illness_types.add(j)
    for i in illness_types:
        train[i] = train.Self_declaration_of_Illness_Form.copy()
    for i in illness_types:
        txt = i+"+"
        regex_pat = re.compile(txt)
        train[i].replace(to_replace=regex_pat, value=float(1), inplace=True, regex=True)
        train[i][(train[i] != float(1))] = '0'
        train[i] =train[i].astype('float64')
    train = train.drop(['Self_declaration_of_Illness_Form'], axis=1)
    train.Virus = train.Virus.astype('object')
    train.Virus[(train.Virus == 'covid')] = '1'
    train.Virus[(train.Virus != '1')] = '0'
    # train.Virus = train.Virus.astype('category')

    # We are keeping the following line of code in case diffrentiating between not covid and other illnesses is important:
    # train.Virus[(train.Virus != 'covid') & (train.Virus != 'not_detected')] = 'other'

    # next line is for evaluating NAN in columns, were used mainly for JOB.
    # NAN_checker(data)

    # removing data as discussed in attached file
    train = train.drop(['ID', 'Address', 'CurrentLocation', 'Job'], axis=1)

    # Part 2.7: discussion is in the attached file
    # we'd like to find the amount of missing data in each feature:
    NAN_checker(train)
    train = train.drop(['PCR_11', 'PCR_15'], axis=1)

    # checking how many jobs we have
    job_col = dataset.Job
    unique_jobs = job_col.unique()

    #This is for part 13, we'd like to see plots before filling missing data. Explanations is in the attached file.

    """""
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    g1 = sns.jointplot(data=train, x="ConversatiosPerDay", y="HappinessScore", hue="Virus")
    g1.ax_joint.grid()
    plt.suptitle('HappinessScore vs. Conversations Per Day')
    plt.show()
    """""

    """""
    g2 = sns.jointplot(data=train, x="AgeGroup", y="NrCousins", hue="Virus")
    g2.ax_joint.grid()
    g2.fig.tight_layout()
    g2.fig.subplots_adjust(top=0.95)  # Reduce plot to make room
    plt.suptitle('Nr. of cousins vs. Age Group')
    plt.savefig('agegroupVScousins.jpg', bbox_inches='tight')
    """""

    """""
    g3 = sns.jointplot(data=train, x="AgeGroup", y="StepsPerYear", hue="Virus")
    g3.ax_joint.grid()
    g3.fig.tight_layout()
    g3.fig.subplots_adjust(top=0.95)  # Reduce plot to make room
    plt.suptitle('Steps Per Year vs. Age Group')
    plt.savefig('agegroupVSsteps.jpg', bbox_inches='tight')
    """""

    """""
    g4 = sns.jointplot(data=train, x="ConversatiosPerDay", y="HouseholdExpenseOnPresents", hue="Virus")
    g4.ax_joint.grid()
    plt.show()
    """""

    """""
    g5 = sns.jointplot(data=train, x="HappinessScore", y="HouseholdExpenseOnPresents", hue="Virus")
    g5.ax_joint.grid()
    plt.show()
    """""

    """""
    g6 = sns.jointplot(data=train, x="DisciplineScore", y="MedicalCarePerYear", hue="Virus")
    g6.ax_joint.grid()
    plt.show()
    """""

    """""
    g7 = sns.jointplot(data=train, x="SocialActivitiesPerDay", y="HouseholdExpenseOnSocialGames", hue="Virus")
    g7.ax_joint.grid()
    plt.show()
    """""

    """""
    g8 = sns.jointplot(data=train, x="SportsPerDay", y="HouseholdExpenseOnSocialGames", hue="Virus")
    g8.ax_joint.grid()
    plt.show()
    """""

    """""
    g9 = sns.jointplot(data=train, x="SocialActivitiesPerDay", y="SportsPerDay", hue="Virus")
    g9.ax_joint.grid()
    # plt.savefig('Socialvssports_noOutliers.jpg', bbox_inches='tight')
    plt.show()
    """""

    """""
    g10_1 = sns.jointplot(data=train, x="StudingPerDay", y="HouseholdExpenseParkingTicketsPerYear")
    g10_1.ax_joint.grid()
    g10_1.fig.tight_layout()
    g10_1.fig.subplots_adjust(top=0.92)  # Reduce plot to make room
    plt.suptitle('Household Expense on Parking Tickets Per Year vs. Studying Per Day \n No Imputation')
    plt.savefig('ticketsVSstudying_noOutliers.jpg', bbox_inches='tight')
    plt.close()
    plt.show()
    """""

    """""
    sns.histplot(train.MedicalCarePerYear, bins=100, kde=True)
    plt.grid()
    plt.suptitle('Medical Care Per Year Distribution')
    plt.savefig('MedicalCare_histogram.png', bbox_inches='tight')
    plt.close()
    """""

    # Part 2.8: Discussion in attached file
    # First, we'd like to change the data into numeric data for easier handling and later modelling.
    col = list(train)

    for i in (range(len(col))):
        if col[i] == 'DateOfPCRTest':
            DateOfBase = train.DateOfPCRTest
            minDate = DateOfBase.min()
            avg_date = (minDate + (DateOfBase - minDate).mean())
            train.DateOfPCRTest = train.DateOfPCRTest.fillna(avg_date)
            for i in train.DateOfPCRTest:
                train.DateOfPCRTest.replace(to_replace=i, value=(i-minDate).days, inplace=True)
            npdata = np.asarray(train.DateOfPCRTest)
            b = stats.zscore(npdata)
            train.DateOfPCRTest= b
        else:
            train[col[i]] = replace_to_mean(train[col[i]])

    # Part 2.9

    """""
    BMI_boxplot = pd.DataFrame(train.StepsPerYear, columns=['StepsPerYear'])
    BMI_boxplot.plot.box(grid='True')
    plt.title('Steps Per Year Box Plot')
    plt.savefig('Steps_boxplot.jpg', bbox_inches='tight')
    plt.close()
    sns.histplot(train.StepsPerYear, bins=10, kde=True)
    plt.grid()
    plt.suptitle('Steps Per Year Distribution')
    plt.savefig('StepsHistogram.png', bbox_inches='tight')
    plt.close()
    """""

    # Part 2.10

    train = remove_outliers(train, col)


    # testing two features . checking if we can drop one of them TODO: Is it duplication or do we need it?
    # g1 = sns.jointplot(data=train, x="ConversatiosPerDay", y="HappinessScore", hue="Virus")
    # plt.show()
    #
    # g2 = sns.jointplot(data=train, x="AgeGroup", y="NrCousins", hue="Virus")
    # plt.savefig('agegroupVScousins.jpg', bbox_inches='tight')
    #
    # g3 = sns.jointplot(data=train, x="AgeGroup", y="StepsPerYear", hue="Virus")
    # plt.savefig('agegroupVSsteps.jpg', bbox_inches='tight')
    #
    # g4 = sns.jointplot(data=train, x="ConversatiosPerDay", y="HouseholdExpenseOnPresents", hue="Virus")
    # plt.show()
    #
    # g5 = sns.jointplot(data=train, x="HappinessScore", y="HouseholdExpenseOnPresents", hue="Virus")
    # plt.show()
    #
    # g6 = sns.jointplot(data=train, x="DisciplineScore", y="MedicalCarePerYear", hue="Virus")
    # plt.show()
    #
    # g7 = sns.jointplot(data=train, x="SocialActivitiesPerDay", y="HouseholdExpenseOnSocialGames", hue="Virus")
    # plt.show()

    # g8 = sns.jointplot(data=train, x="SportsPerDay", y="HouseholdExpenseOnSocialGames", hue="Virus")
    # plt.show()

    #g9 = sns.jointplot(data=train, x="SocialActivitiesPerDay", y="SportsPerDay", hue="Virus")
    #plt.show()

    # g10_2 = sns.jointplot(data=train, x="StudingPerDay", y="HouseholdExpenseParkingTicketsPerYear")
    # g10_2.ax_joint.grid()
    # g10_2.fig.tight_layout()
    # g10_2.fig.subplots_adjust(top=0.92)  # Reduce plot to make room
    # plt.suptitle('Household Expense on Parking Tickets Per Year vs. Studying Per Day \n After Imputation')
    # plt.savefig('ticketsVSstudying.jpg', bbox_inches='tight')
    # plt.close()
    # plt.show()

    # Part 3.12
    train = convertToNumerical(train)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    #
    # ax1.scatter(train.X, train.Virus, s=10, c='b', marker="s", label='first')
    # ax1.scatter(train.Y, train.Virus, s=10, c='r', marker="o", label='second')
    # plt.legend(loc='upper left')
    # plt.show()
    #
    # plt.scatter(train.DateOfPCRTest, train.Virus)
    # plt.show()
    # plt.scatter(train.StepsPerYear, train.Virus)
    # plt.show()
    # plt.scatter(train.PCR_95, train.Virus)
    # plt.show()
    # plt.scatter(train.PCR_10, train.Virus)
    # plt.show()
    # plt.scatter(train.X, train.Virus)
    # plt.show()
    # plt.scatter(train.Y, train.Virus)
    # plt.show()
    #
    # plt.scatter(train.DateOfPCRTest, train.Risk)
    # plt.show()
    # plt.scatter(train.StepsPerYear, train.Risk)
    # plt.show()
    # plt.scatter(train.PCR_95, train.Risk)
    # plt.show()
    # plt.scatter(train.PCR_10, train.Risk)
    # plt.show()
    #
    # plt.scatter(train.DateOfPCRTest, train.SpreadLevel)
    # plt.show()
    # plt.scatter(train.StepsPerYear, train.SpreadLevel)
    # plt.show()
    # plt.scatter(train.PCR_95, train.SpreadLevel)
    # plt.show()
    # plt.scatter(train.PCR_10, train.SpreadLevel)
    # plt.show()

    # Part 3.13
    # Illness types corralations matrix:
    """""
    illness_types = list(illness_types)
    train["Virus"] = train["Virus"].astype('float64')
    illness_types.append("Virus")
    illness_types.append("SpreadLevel")
    illness_types.append("Risk")
    illness_df = train[illness_types]
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    corrMatrix = illness_df.corr()
    plt.figure(figsize=(20, 20))
    plt.title('Symptoms Correlation Map', fontsize=20)
    ax = sns.heatmap(corrMatrix, xticklabels=True, yticklabels=True, annot=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # plt.show()
    plt.savefig('correlation_matrix2.jpg', bbox_inches='tight')
    """""

    # ID3 for Virus target label
    """""
    X_train = train.iloc[:,indexes]
    Y_train = train.Virus
    h = DecisionTreeClassifier(criterion="entropy", max_depth=15)
    h.fit(X_train, Y_train)
    # plt.figure(figsize=(6, 6))
    # plot_tree(h, filled=True)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=130)
    plot_tree(h, filled=True)
    plt.show()
    """""

    #ID3 for SpreadLevel target label
    """""
    X_train = train.iloc[:,indexes]
    Y_train = train.SpreadLevel
    h = DecisionTreeClassifier(criterion="entropy", max_depth=15)
    h.fit(X_train, Y_train)
    # plt.figure(figsize=(6, 6))
    # plot_tree(h, filled=True)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=130)
    plot_tree(h, filled=True)
    plt.show()
 """""

    # ID3 for Risk target label
    """""
    X_train = train.iloc[:,indexes]
    Y_train = train.Risk
    h = DecisionTreeClassifier(criterion="entropy", max_depth=15)
    h.fit(X_train, Y_train)
    # plt.figure(figsize=(6, 6))
    # plot_tree(h, filled=True)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=130)
    plot_tree(h, filled=True)
    plt.show()
    """""

    # Part 13:

    # all corralations matrix
    # chosen features:
    indexes = list(range(0, 38))
    indexes.append(38)
    indexes.append(45)
    indexes.append(49)

    """""
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    corrMatrix = train.corr()
    plt.figure(figsize=(20, 20))
    plt.title('Final Correlation Map', fontsize=20)
    ax = sns.heatmap(corrMatrix, xticklabels=True, yticklabels=True, annot=True)
    plt.show()
    # plt.savefig('correlation_matrix.jpg', bbox_inches='tight')
     """""
    # # All plots required for this assignment were made here: #TODO: is this still relevant?
    # # BMI histogram
    # sns.histplot(train.BMI, bins=100, kde=True)
    # plt.grid()
    # plt.savefig('BMI_histogram.png')
    # plt.close()
    #
    # Blood type bins:
    # datasetCopy.Virus[(datasetCopy.Virus != 'covid') & (datasetCopy.Virus != 'not_detected')] = 'other'
    BloodType_plot = pd.crosstab([train.BloodType], train.Virus)
    BloodType_plot.plot(kind='bar', stacked=True, color=['red', 'blue'], grid=True)
    plt.grid()
    plt.savefig('Bloodtype_histogram.png')
    plt.close()

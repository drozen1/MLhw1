import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeClassifier
import warnings
import scipy.stats as stats

warnings.filterwarnings('ignore')


def convertToNumerical(train):
    bloodTypes = pd.Series(train.BloodType).unique()
    count = 0
    for i in bloodTypes:
        train.BloodType.replace(to_replace=i, value=count, inplace=True)
        count += 1
    train.Sex.replace(to_replace="F", value=1, inplace=True)
    train.Sex.replace(to_replace="M", value=0, inplace=True)

    viruses = pd.Series(train.Virus).unique()
    count = 0
    for i in viruses:
        train.Virus.replace(to_replace=i, value=count, inplace=True)
        count += 1
    risks = pd.Series(train.Risk).unique()
    count = 0
    for i in risks:
        train.Risk.replace(to_replace=i, value=count, inplace=True)
        count += 1
    spreadLevels = pd.Series(train.SpreadLevel).unique()
    count = 0
    for i in spreadLevels:
        train.SpreadLevel.replace(to_replace=i, value=count, inplace=True)
        count += 1

    # print(train.info())
    # train.Sex = train.Sex.astype('float64')
    # print(train.info())
    # train.Sex = train.Sex.astype('float64')
    # print(dataset.info())
    # pd.to_numeric(train.Sex, errors='coerce')
    # print(dataset.info())
    # train.Sex = train.Sex.astype('int64')
    # train.BloodType = dataset.BloodType.astype('category')


def remove_outliers(data, columns):
    all_indexes = []
    x = np.zeros(data.shape[0], dtype=bool)
    for i in (range(len(columns))):
        if isinstance(data[columns[i]][0], str):
            pass
        else:
            if (columns[i] != "DateOfPCRTest"):
                npdata = np.asarray(data[columns[i]])
                b = stats.zscore(npdata)
                # if columns[i]=='BMI':
                #     plt.figure(figsize=(20, 20))
                #     BMI_boxplot = pd.DataFrame(b, columns=['BMI'])
                #     BMI_boxplot.plot.box(grid='True')
                #     plt.title('BMI Box Plot after z-score Transformation')
                #     plt.savefig('BMI_boxplot2.jpg', bbox_inches='tight')
                #     plt.close()
                data[columns[i]] = b  # remove this line for generating graphs
                x = ((np.absolute(b) >= 3) | x)
    afterDrop = data[(x == False)]

    # plt.figure(figsize=(20, 20))
    # BMI_boxplot2 = pd.DataFrame(afterDrop.BMI, columns=['BMI'])
    # BMI_boxplot2.plot.box(grid='True')
    # plt.title('BMI Box Plot after z-score Transformation and Outliers Removal')
    # plt.savefig('BMI_boxplot3.jpg', bbox_inches='tight')
    # plt.close()
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
    train.X, train.Y = split_in_XY(train.CurrentLocation)
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
    # TODO remove according to table + show NAN amounts + change to train
    # we'd like to find the amount of missing data in each feature:
    NAN_checker(train)
    train = train.drop(['PCR_11', 'PCR_15'], axis=1)

    # # This is for part 13, we'd like to see plots before filling missing data. Explanations is in the attached file.
    # # plt.rc('xtick', labelsize=20)
    # # plt.rc('ytick', labelsize=20)
    # g1 = sns.jointplot(data=train, x="ConversatiosPerDay", y="HappinessScore", hue="Virus")
    # g1.ax_joint.grid()
    # plt.suptitle('HappinessScore vs. Conversations Per Day')
    # plt.show()
    #
    # g2 = sns.jointplot(data=train, x="AgeGroup", y="NrCousins", hue="Virus")
    # g2.ax_joint.grid()
    # g2.fig.tight_layout()
    # g2.fig.subplots_adjust(top=0.95)  # Reduce plot to make room
    # plt.suptitle('Nr. of cousins vs. Age Group')
    # plt.savefig('agegroupVScousins.jpg', bbox_inches='tight')
    #
    # g3 = sns.jointplot(data=train, x="AgeGroup", y="StepsPerYear", hue="Virus")
    # g3.ax_joint.grid()
    # g3.fig.tight_layout()
    # g3.fig.subplots_adjust(top=0.95)  # Reduce plot to make room
    # plt.suptitle('Steps Per Year vs. Age Group')
    # plt.savefig('agegroupVSsteps.jpg', bbox_inches='tight')
    #
    # g4 = sns.jointplot(data=train, x="ConversatiosPerDay", y="HouseholdExpenseOnPresents", hue="Virus")
    # g4.ax_joint.grid()
    # plt.show()
    #
    # g5 = sns.jointplot(data=train, x="HappinessScore", y="HouseholdExpenseOnPresents", hue="Virus")
    # g5.ax_joint.grid()
    # plt.show()
    #
    # g6 = sns.jointplot(data=train, x="DisciplineScore", y="MedicalCarePerYear", hue="Virus")
    # g6.ax_joint.grid()
    # plt.show()
    #
    # g7 = sns.jointplot(data=train, x="SocialActivitiesPerDay", y="HouseholdExpenseOnSocialGames", hue="Virus")
    # g7.ax_joint.grid()
    # plt.show()
    #
    # g8 = sns.jointplot(data=train, x="SportsPerDay", y="HouseholdExpenseOnSocialGames", hue="Virus")
    # g8.ax_joint.grid()
    # plt.show()
    #
    # g9 = sns.jointplot(data=train, x="SocialActivitiesPerDay", y="SportsPerDay", hue="Virus")
    # g9.ax_joint.grid()
    # # plt.savefig('Socialvssports_noOutliers.jpg', bbox_inches='tight')
    # plt.show()
    #
    # g10_1 = sns.jointplot(data=train, x="StudingPerDay", y="HouseholdExpenseParkingTicketsPerYear")
    # g10_1.ax_joint.grid()
    # g10_1.fig.tight_layout()
    # g10_1.fig.subplots_adjust(top=0.92)  # Reduce plot to make room
    # plt.suptitle('Household Expense on Parking Tickets Per Year vs. Studying Per Day \n No Imputation')
    # plt.savefig('ticketsVSstudying_noOutliers.jpg', bbox_inches='tight')
    # plt.close()
    # # plt.show()

    # sns.histplot(train.MedicalCarePerYear, bins=100, kde=True)
    # plt.grid()
    # plt.suptitle('Medical Care Per Year Distribution')
    # plt.savefig('MedicalCare_histogram.png', bbox_inches='tight')
    # plt.close()

    # Part 2.8: Discussion in attached file
    # First, we'd like to change the data into numeric data for easier handling and later modelling.
    col = list(train)

    for i in (range(len(col))):
        if col[i] == 'DateOfPCRTest':
            DateOfBase = train.DateOfPCRTest
            minDate = DateOfBase.min()
            avg_date = (minDate + (DateOfBase - minDate).mean())
            train.DateOfPCRTest = train.DateOfPCRTest.fillna(avg_date)

        else:
            train[col[i]] = replace_to_mean(train[col[i]])

    # Part 2.9

    # plt.figure(figsize=(20, 20))
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

    # Part 2.10

    train = remove_outliers(train, col)
    print(train.info())

    # train= convertToNumerical(train)

    # Part 13:

    # all corralations matrix
    # plt.rc('xtick', labelsize=20)
    # plt.rc('ytick', labelsize=20)
    # corrMatrix = train.corr()
    # plt.figure(figsize=(20, 20))
    # ax = sns.heatmap(corrMatrix, xticklabels=True, yticklabels=True, annot=True)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    # # plt.show()
    # plt.savefig('correlation_matrix.jpg', bbox_inches='tight')

    # # testing two features . checking if we can drop one of them
    # g1 = sns.jointplot(data=train, x="ConversatiosPerDay", y="HappinessScore", hue="Virus")
    # plt.show()
    #
    # g2 = sns.jointplot(data=train, x="AgeGroup", y="NrCousins", hue="Virus")
    # plt.savefig('agegroupVScousins.jpg', bbox_inches='tight')
    #
    # g3 = sns.jointplot(data=train, x="AgeGroup", y="StepsPerYear", hue="Virus")
    # plt.savefig('agegroupVSsteps.jpg', bbox_inches='tight')

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
    #
    # g8 = sns.jointplot(data=train, x="SportsPerDay", y="HouseholdExpenseOnSocialGames", hue="Virus")
    # plt.show()
    #
    # g9 = sns.jointplot(data=train, x="SocialActivitiesPerDay", y="SportsPerDay", hue="Virus")
    # plt.show()
    #
    g10_2 = sns.jointplot(data=train, x="StudingPerDay", y="HouseholdExpenseParkingTicketsPerYear")
    g10_2.ax_joint.grid()
    g10_2.fig.tight_layout()
    g10_2.fig.subplots_adjust(top=0.92)  # Reduce plot to make room
    plt.suptitle('Household Expense on Parking Tickets Per Year vs. Studying Per Day \n After Imputation')
    plt.savefig('ticketsVSstudying.jpg', bbox_inches='tight')
    plt.close()
    # plt.show()

    # X_train = train.iloc[:,[0, 33]]
    # Y_train = train.Virus
    # h = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    # h.fit(X_train, Y_train)
    # # plt.figure(figsize=(6, 6))
    # # plot_tree(h, filled=True)
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=130)
    # plot_tree(h, filled=True)
    #
    # plt.show()

    # # All plots required for this assignment were made here:
    # # BMI histogram
    # sns.histplot(train.BMI, bins=100, kde=True)
    # plt.grid()
    # plt.savefig('BMI_histogram.png')
    # plt.close()
    #
    # # Blood type bins:
    # datasetCopy.Virus[(datasetCopy.Virus != 'covid') & (datasetCopy.Virus != 'not_detected')] = 'other'
    # BloodType_plot = pd.crosstab([datasetCopy.BloodType], datasetCopy.Virus)
    # BloodType_plot.plot(kind='bar', stacked=True, color=['red', 'blue', 'green'], grid=True)
    # plt.grid()
    # plt.savefig('Bloodtype_histogram.png')
    # plt.close()


import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import IsolationForest
import math
import scipy.stats as stats


# Press the green button in the gutter to run the script.

def update_BMI_outliers(dataset):
    return dataset[dataset.BMI<40]

def feature_graph(dataset_type):
    _ = sns.histplot(dataset_type, kde=True)  # prettier with seaborn
    plt.grid()
    plt.show()

def clean_table(table):
    lst = []
    for i in range(table.shape[0]):
        if (math.isnan(table.BMI.values[i]) == False):
            if (dataset.Virus[i] == "covid"):
                lst.append([table.BMI.values[i], 1])
            else:
                lst.append([table.BMI.values[i], 0])
    return lst

def clear_table(table,table_type):
    lst = []
    for i in range(table.shape[0]):
        if (math.isnan(table_type.values[i]) == False):
            lst.append(table_type.values[i])
    return lst


def count_nans(dataset_type):
    lst = []
    counter=0
    for i in range(dataset_type.shape[0]):
        if (math.isnan(dataset_type.values[i]) ):
            counter+=1
    print(counter/dataset_type.shape[0])

# def who_nan(dataset):





def isolation_forest(train_lst,test_lst,validate_lst):
    # # fit the model
    train_np = np.asarray(train_lst)
    test_np = np.asarray(test_lst)
    validate_np = np.asarray(validate_lst)

    # fit the model
    clf = IsolationForest(max_samples=100)
    clf.fit(train_np)
    y_pred_train = clf.predict(train_np)
    y_pred_test = clf.predict(test_np)
    y_pred_outliers = clf.predict(validate_np)

    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    b1 = plt.scatter(train_np[:, 0], train_np[:, 1], c='white',
                     s=20, edgecolor='k')
    b2 = plt.scatter(test_np[:, 0], test_np[:, 1], c='green',
                     s=20, edgecolor='k')
    c = plt.scatter(validate_np[:, 0], validate_np[:, 1], c='red',
                    s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((0, 30))
    plt.ylim((-0.3, 1.3))
    plt.legend([b1, b2, c],
               ["training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left")
    plt.show()

def remove_outliers(dataset,dataset_type):
    dataclean = clear_table(dataset, dataset_type)
    npdata = np.asarray(dataclean)
    b = stats.zscore(npdata)




if __name__ == '__main__':

    dataset=pd.read_csv('virus_hw1.csv')

    #count_nans(dataset.PCR_11)
    dataset.DateOfPCRTest=dataset.DateOfPCRTest.astype("datetime64")
    dataset=dataset.drop(['ID'], axis=1)
    print(dataset.info())

    dob = dataset.DateOfPCRTest
    m = dob.min()
    avg_date = (m + (dob - m).mean()).to_pydatetime()

    #dataset.DateOfPCRTest.values[0].astype('float')
    # min(dataset.DateOfPCRTest).astype('float')

    train, test_temp = train_test_split(dataset, test_size=0.4, random_state=14)
    test, validate = train_test_split(test_temp, test_size=0.5, random_state=14)

    test_lst=clean_table(test)
    train_lst = clean_table(train)
    validate_lst = clean_table(validate)

    check_null = dataset.isnull()
    columns = list(check_null)
    percent = [0] * (len(columns))
    for i in (range(len(columns))):
        check = check_null[columns[i]]
        percent[i] = columns[i]
        # check.value_counts(normalize=True)

    # with open('VariableTypes.csv', 'w', newline='') as csvfile:
    #     my_writer = csv.writer(csvfile, delimiter=' ' , dialect='excel')
    #     for i in percent:
    #         my_writer.writerow(i)

    # count_nans(dataset.sex)

    isolation_forest(test_lst,train_lst,validate_lst)

    for column in dataset:
        print(dataset[column])
        #transfer to z-score
    dataclean = clear_table(dataset,dataset.BMI)
    npdata =np.asarray(dataclean)
    # npdata= data.to_numpy()
    npdata=stats.zscore(npdata)

   # print(feature_graph(dataset.BMI)) ##remove <40
    print(feature_graph(dataset.ConversatiosPerDay)) #good
    print(feature_graph(dataset.DisciplineScore ))# remove <=10
    print(feature_graph(dataset.AgeGroup)) #good
    print(feature_graph(dataset.HouseholdExpenseOnPresents))
    print(feature_graph(dataset.HouseholdExpenseOnSocialGames))
    print(feature_graph(dataset.HouseholdExpenseParkingTicketsPerYear))
    print(feature_graph(dataset.MedicalCarePerYear))
    print(feature_graph(dataset.NrCousins))
    count_nans(dataset.PCR_10)
    print(feature_graph(dataset.PCR_10)) # here you have some
    count_nans(dataset.PCR_11)
    print(feature_graph(dataset.PCR_11)) #lots of nan value 83%
    count_nans(dataset.PCR_15)
    print(feature_graph(dataset.PCR_15)) #lots of nan value 83%
    count_nans(dataset.PCR_17)
    print(feature_graph(dataset.PCR_17)) #nan value 10% here you have some
    count_nans(dataset.PCR_19)
    print(feature_graph(dataset.PCR_19)) #nan value 9% not sure
    count_nans(dataset.PCR_32)
    print(feature_graph(dataset.PCR_32))  #nan value 10% here you have some
    count_nans(dataset.PCR_45)
    print(feature_graph(dataset.PCR_45)) #nan value 10% here you have some
    count_nans(dataset.PCR_46)
    print(feature_graph(dataset.PCR_46)) #nan value 10% here you have some
    count_nans(dataset.PCR_7)
    print(feature_graph(dataset.PCR_7)) #nan value 9% not sure
    count_nans(dataset.PCR_72)
    print(feature_graph(dataset.PCR_72)) #nan value 9% not sure
    count_nans(dataset.PCR_76)
    print(feature_graph(dataset.PCR_76))  #nan value 10% here you have some
    count_nans(dataset.PCR_8)
    print(feature_graph(dataset.PCR_8)) #nan value 10% here you have some
    count_nans(dataset.PCR_83)
    print(feature_graph(dataset.PCR_83)) #nan value 10% here you have some
    count_nans(dataset.PCR_89)
    print(feature_graph(dataset.PCR_89)) #nan value 10% here you have some
    count_nans(dataset.PCR_9)
    print(feature_graph(dataset.PCR_9)) #nan value 10% here you have some
    count_nans(dataset.PCR_93)
    print(feature_graph(dataset.PCR_93))#nan value 10% here you have some
    count_nans(dataset.PCR_95)
    print(feature_graph(dataset.PCR_95)) #nan value 9% not sure
    print(dataset.info())




    # plt.plot([1, 2, 3, 4])
    # plt.ylabel('some numbers')
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

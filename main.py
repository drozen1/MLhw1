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

# Press the green button in the gutter to run the script.

def update_BMI_outliers(dataset):
    return dataset[dataset.BMI<40]

def BMI_graph(dataset):
    dataset = pd.read_csv('virus_hw1.csv')
    _ = sns.histplot(dataset.BMI, kde=True)  # prettier with seaborn
    plt.grid()
    plt.show()

if __name__ == '__main__':

    dataset=pd.read_csv('virus_hw1.csv')



    lst=[]
    for i in range(3000):
        if(math.isnan(dataset.BMI[i])==False):
            if(dataset.Virus[i]=="covid"):
                lst.append([dataset.BMI[i],1])
            else:
                lst.append([dataset.BMI[i], 0])





    # # fit the model
    # lst=np.asarray(lst)
    # clf = IsolationForest(max_samples=100)
    # clf.fit(lst)
    # y_pred_train = clf.predict(lst)
    #
    # # plot the line, the samples, and the nearest vectors to the plane
    # xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    # Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    #
    # plt.title("IsolationForest")
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    #
    # b1 = plt.scatter(lst[:, 0], lst[:, 1], c='white',
    #                  s=20, edgecolor='k')
    #
    # plt.axis('tight')
    # plt.xlim((0, 30))
    # plt.ylim((0, 2))
    # plt.legend([b1],
    #            ["training observations"],
    #            loc="upper left")
    # plt.show()
    #
    #
    #




    # plt.plot([1, 2, 3, 4])
    # plt.ylabel('some numbers')
    # plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

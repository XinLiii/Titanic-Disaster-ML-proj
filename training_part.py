from data_processing import data_proc
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm

filling_method = int(sys.argv[1])
training_method = sys.argv[2]

def split_data(data_frame):
    x_df,y_df =data_frame.loc[:,data_frame.columns != 'Survived'],data_frame['Survived']
    x_train, x_valid, y_train, y_valid = train_test_split(x_df, y_df, test_size=0.2, random_state=100)
    # the random_state: if do not give any value to random_state, every time we run it will get a different result
    # if we specify it with a value, we will get the same result, but different values means different random result
    return x_train, x_valid, y_train, y_valid

def data_normalize(x_train_df,x_valid_df,x_test_df):
    scaler = MinMaxScaler()
    x_train_df_mms, x_valid_df_mms, x_test_df_mms = scaler.fit_transform(x_train_df),scaler.transform(x_valid_df),\
                                                    scaler.transform(x_test_df)
    return x_train_df_mms, x_valid_df_mms, x_test_df_mms


def train_mod(x_train_mms,x_valid_mms,y_train, method_type):

    if method_type == 'lr':
        lr = LogisticRegression(C=1)
        lr.fit(x_train_mms, y_train)
        test_result = lr.predict(x_valid_mms)
        result_value = np.array(test_result).reshape(-1)
        return result_value, lr

    elif method_type == 'svm':
        svm_c = svm.SVC(kernel='rbf', gamma=20, decision_function_shape='ovr')
        svm_c.fit(x_train_mms, y_train)
        test_result = svm_c.predict(x_valid_mms)
        result_value = np.array(test_result).reshape(-1)
        return result_value, svm_c
    return

def accuracy(real_v,result_v):
    result_dic = {'real value': real_v, 'predict value': result_v}
    result_df = pd.DataFrame(data=result_dic)
    result_df['result'] = result_df['real value'] ^ result_df['predict value']
    tmp_sum = sum(result_df['result'])
    accuracy_of_test = 1 - tmp_sum / len(result_df['result'])
    print(accuracy_of_test)


get_data = data_proc("./raw_data/train.csv", "./raw_data/test.csv")

#get_data.raw_describe()
#print('\n')
#get_data.missing_data_showing()

train_df, test_df = get_data.total_data_proc(2)
x_tr, x_val, y_tr, y_val = split_data(train_df)
x_tr_mms, x_val_mms, x_test_mms = data_normalize(x_tr, x_val, test_df)
real_v = np.array(y_val).reshape(-1)
result_v, modl = train_mod(x_tr_mms,x_val_mms,y_tr,'svm')
accuracy(real_v, result_v)

test_result = modl.predict(x_test_mms)
print(test_result)
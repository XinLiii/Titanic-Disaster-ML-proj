import numpy as np
import pandas as pd



class data_proc(object):
    def __init__(self, training_file, testing_file):
        self.training_file = pd.read_csv(training_file, header=0)
        self.testing_file = pd.read_csv(testing_file, header=0)
        self.name_title_dic = {}

    #def __read_raw_data(self):
     #   return pd.read_csv(self.training_file, header=0), pd.read_csv(self.testing_file, header=0)

    """
    To see the content in raw data
    """
    def raw_describe(self, num_show=5):
        print("Content in raw data: \n")
        print(self.training_file.head(num_show),"\n")
        print("Data describe: \n")
        print(self.training_file.describe(),'\n')


    """
    Show the number of missing value of each feature 
    """
    def missing_data_showing(self):
        train_missing_data = self.training_file.isnull().sum().sort_values(ascending=False)
        test_missing_data = self.testing_file.isnull().sum().sort_values(ascending=False)
        print('Missing data in training file: ')
        print(train_missing_data,'\n')
        print('Missing data in testing file: ')
        print(test_missing_data)

    """
    for each name, there exits the title for one person, use these titles as a new feature,
    and change it into numbers
    """
    def __get_person_title(self, name_str):
        split_part = name_str.split(sep=' ')
        for item in split_part:
            if '.' in item:
                return item
        return

    """
    this method is to build the title dictionary and the new title feature sequence
    """
    def __title_feature(self, data_frame):
        new_title_feature_list = []
        name_title_dic = {}
        for name_item in data_frame['Name']:
            title_of_item = self.__get_person_title(name_item)
            if name_title_dic.get(title_of_item, -1) == -1:
                name_title_dic[title_of_item] = 1
            else:
                name_title_dic[title_of_item] += 1
            new_title_feature_list.append(title_of_item)
        #print(name_title_dic)
        return np.array(new_title_feature_list).reshape(-1,1)

    """
    test for title feature method
    """
    def print_title_feature(self, data_frame):
        print(self.__title_feature(data_frame))


    """
    change 'male' and 'female' in Sex into 1 and 0, 1 means 'male', 0 means 'female'
    """
    def __sex_feature(self, data_frame):
        new_sex_feature_list = []
        for sex_item in data_frame['Sex']:
            if sex_item == 'male':
                new_sex_feature_list.append(1)
            elif sex_item == 'female':
                new_sex_feature_list.append(0)
        return np.array(new_sex_feature_list)

    """
    filling gaps in a feature, there should be several methods to achieve this
    mt:
    '0' -- drop vacant value
    '1' -- using mean value
    '2' -- using median value
    '3' -- using a previous value
    '4' -- using a next value
    '5' -- using the average age of corresponding title
    """
    def __fill_feature(self, data_frame, feature_name, mt):
        age_mean_value = data_frame[feature_name].mean()
        age_median_value = data_frame[feature_name].median()
        if mt == 0:
            self.__rm_vac_value(data_frame, feature_name)
        elif mt == 1:
            data_frame[feature_name].fillna(age_mean_value, inplace=True)
        elif mt == 2:
            data_frame[feature_name].fillna(age_median_value, inplace=True)
        elif mt == 3:
            data_frame[feature_name].fillna(method='pad', inplace=True)
        elif mt == 4:
            data_frame[feature_name].fillna(method='bfill', inplace=True)
        elif mt == 5:
            self.__filling_vac_func(data_frame)

    """
    filling function by using title features
    """
    def __filling_vac_func(self, data_frame):
        title_avg_age = data_frame.groupby('Title of person').mean()['Age']

        title_avg_age_dict = title_avg_age.to_dict()
        data_frame['Title Flag'] = 0
        age_nan = np.isnan(data_frame['Age'])
        age_nan_index = data_frame.loc[age_nan].index
        data_frame.loc[age_nan_index,'Age'] = data_frame.loc[age_nan_index, 'Title of person'].map(title_avg_age_dict)
        data_frame.loc[age_nan_index,'Title Flag'] = 1

    """
    drop cabin feature
    """
    def __drop_one_feature(self, data_frame, feature_name):
        data_frame.drop(columns=feature_name, inplace=True)

    """
    remove the vacant items in Embarked feature 
    """
    def __rm_vac_value(self, data_frame, feature_name, in_place = True):
        data_frame.drop(data_frame[pd.isnull(data_frame[feature_name])].index, inplace=in_place)


    def feature_processing(self, data_frame,mt):
        #train_df, test_df = pd.read_csv(self.training_file, header=0), pd.read_csv(self.testing_file, header=0)
        self.__drop_one_feature(data_frame, 'Ticket')
        self.__drop_one_feature(data_frame, 'Cabin')
        num_col = len(data_frame.columns)
        new_tit_feature = self.__title_feature(data_frame)
        data_frame.insert(num_col, 'Title of person', new_tit_feature)
        self.__drop_one_feature(data_frame, 'Name')
        self.__fill_feature(data_frame,'Age',mt)
        titles_map_dict = {'Capt.': 'Other', 'Major.': 'Other', 'Jonkheer.': 'Other', 'Don.': 'Other',
               'Sir.': 'Other', 'Dr.': 'Other', 'Rev.': 'Other', 'Countess.': 'Other', 'Dona.': 'Other',
               'Mme.': 'Mrs.', 'Mlle.': 'Miss.', 'Ms.': 'Miss.', 'Mr.': 'Mr.', 'Mrs.': 'Mrs.', 'Miss.': 'Miss.',
               'Master.': 'Master.', 'Lady': 'Other'}
        data_frame['Title of person'] = data_frame['Title of person'].map(titles_map_dict)

        data_frame.drop('PassengerId', axis=1, inplace=True)
        data_frame['Sex'] = pd.Categorical(data_frame['Sex'])
        data_frame['Embarked'] = pd.Categorical(data_frame['Embarked'])
        data_frame['Title of person'] = pd.Categorical(data_frame['Title of person'])
        data_frame = pd.get_dummies(data_frame)
        self.__drop_one_feature(data_frame, 'Sex_female')
        self.__drop_one_feature(data_frame, 'Embarked_C')
        self.__drop_one_feature(data_frame, 'Title of person_Master.')
        data_frame['Family size'] = data_frame['SibSp'] + data_frame['Parch']
        self.__drop_one_feature(data_frame, 'SibSp')
        self.__drop_one_feature(data_frame, 'Parch')
        return data_frame


    def total_data_proc(self, mt):
        train_df = self.feature_processing(self.training_file, mt)
        test_df = self.feature_processing(self.testing_file, mt)
        #print(train_df.head(8))
        return train_df, test_df

    def __feature_extra(self, data_frame):
        header = data_frame.head(0)
        feature_list = []
        for feature in header:
            feature_list.append(feature)


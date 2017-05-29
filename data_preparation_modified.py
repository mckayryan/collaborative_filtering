import random
import numpy as npy
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold


random.seed()

class dataset(object):

    def __init__(self, path, dataset):
        self.raw_df = pd.DataFrame(pd.read_csv(path + '%s.csv' % dataset))

        self.item_index = dict(
            zip(
                sorted(self.raw_df.movieId.unique()),
                range(len(self.raw_df.movieId.unique()))
                )
            )

        self.master_df = self.load_master_df()

        self.train_sets = [ data_dict() ]
        self.test_sets = [ data_dict() ]



    def load_master_df(self, index='movieId', columns='userId', values='rating'):
        pivot_ratings = self.raw_df.pivot(index=index, columns=columns, values=values).as_matrix()

        return data_dict(
                rating = npy.nan_to_num(pivot_ratings),
                rated = ~npy.isnan(pivot_ratings),
                index = npy.array(list(range(len(self.raw_df))))
            )

    def df_from_index_list(self, index_list, df_size):

        df = npy.zeros(df_size)

        for index, data_index in enumerate(index_list):
            try:
                df[
                    self.item_index[
                        self.raw_df['movieId'].loc[data_index]],
                        self.raw_df['userId'].loc[data_index]-1
                ] = self.raw_df['rating'].loc[data_index]
            except IndexError:
                print('(', self.item_index[self.raw_df['movieId'].loc[data_index]],    ', ', self.raw_df['userId'].loc[data_index],
                                ')\n')

        return  data_dict(
                    rating = df,
                    rated = df.astype(bool),
                    index = npy.array(index_list)
                )


    def random_sample_split(self, dataset_index=None, test_split=0.2):
        if dataset_index == None: dataset_index = len(self.train_sets)

        test_ss = ShuffleSplit(n_splits=1, test_size=test_split)

        ''' Train and Test are disjoint partitions '''

        for train, test in test_ss.split(self.raw_df):
            '''
                Append to test_sets
                test ratings matrix: [ index(movieID), userID ]
                test rated bool matrix: True when ratings > 1, False otherwise
            '''
            self.test_sets.insert(
                dataset_index,
                self.df_from_index_list(
                    index_list=test,
                    df_size=self.master_df['size']
                )
            )


            '''
                Append to train_sets
                test ratings matrix: [ index(movieID), userID ]
                test rated bool matrix: True when ratings > 1, False otherwise
            '''
            self.train_sets.insert(
                dataset_index,
                self.df_from_index_list(
                    index_list=train,
                    df_size=self.master_df['size']
                )
            )

        print_df_details(self.train_sets[dataset_index]['rating'],'test_df')
        print_df_details(self.test_sets[dataset_index]['rating'],'train_df')

        return dataset_index

def kfold_validation_generator(data, kfolds=6, randomise=True):
    """
        Takes data and generates k training, validation pairs
        training and validation sets are disjoint sets of lengths len(data)/k (training) and (k-1)*len(data)/k (validation)
    """

    kfold_split = KFold(n_splits=kfolds, shuffle=randomise)

    for train_index, validation_index in kfold_split.split(data):
        yield train_index, validation_index
        # print(train_index, validation_index)


def data_dict(rating=[], rated=[], index=[]):
    return { 'rating': rating, 'rated': rated, 'size': npy.array(rating).shape, 'index': index}

def print_df_details(df, name):
    print('***', name, '***', 'Shape: ', df.shape)

# sampling the data





# splitting the data


# evaluation metrics

def main():
    d = dataset(path='./ml-latest-small/', dataset='ratings')
    index = d.random_sample_split()
    print(d.train_sets[index]['size'])
    # print (     pd.DataFrame(d.train_sets[index]['rating'])
    #             .melt()
    #             .nlargest(20, 'variable')
    #     )
    # for i in range(6):
    #     # t, v = kfold_validation_generator(d.train_sets[0]['rating'])
        # print len(len(t), len(v)

    t_v_generator = kfold_validation_generator(d.train_sets[index]['index'], kfolds=6)
    for i, j in t_v_generator:
        train_index = d.train_sets[index]['index'][i]
        validation_index = d.train_sets[index]['index'][j]
        # then do something with this fold

if __name__ == '__main__':
    import sys
    main()
import random
import numpy as npy
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize
# from train_X_Theta_v1 import *

random.seed()

num_features = 105
lamb = 0.17
num_kfold = 5
partial = 0.15
debug_level = 1

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

        self.train_sets = [ ]
        self.test_sets = [ ]
        self.analysis_sets = [ ]



    def load_master_df(self, index='movieId', columns='userId', values='rating'):

        pivot_ratings = self.raw_df.pivot(index=index, columns=columns, values=values).as_matrix()

        return self.data_dict(
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

        return  self.data_dict(
                    rating = df,
                    rated = df.astype(bool),
                    index = npy.array(index_list)
                )

    def temporal_split(self, dataset_index=None, test_split=0.2):
        '''
            temporal_split
            Orders a dataset by 'timestamp' and partitions the most recent len(dataset)*test_split userId's as a test set and the reamaining len(dataset)*(1-test_split) as a training set
            These sets are appended to the train_sets, test_sets array
            Returns index of partitioned sets in train_sets, test_sets
        '''
        if dataset_index == None: dataset_index = len(self.train_sets)

        temporal_sorted = self.raw_df.sort_values(by='timestamp', ascending=False)

        test_df = temporal_sorted.nlargest(int(len(temporal_sorted)*(test_split)), 'timestamp')
        test_index_list = test_df['userId']

        # Train and Test are disjoint partitions
        train_index_list = set(temporal_sorted['userId']) - set(test_index_list)
        '''
            Append to test_sets
            test ratings matrix: [ index(movieID), userID ]
            test rated bool matrix: True when ratings > 1, False otherwise
        '''
        self.test_sets.insert(
            dataset_index,
            self.df_from_index_list(
                index_list=test_index_list,
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
                index_list=train_index_list,
                df_size=self.master_df['size']
            )
        )
        if debug_level > 1:
            print_df_details(self.train_sets[dataset_index]['rating'],'temporal test_df')
            print_df_details(self.test_sets[dataset_index]['rating'],'temporal train_df')

        return dataset_index

    def random_sample_split(self, dataset_index=None, test_split=0.2):
        '''
            random_sample_split
            randomly samples and partitions raw_df into a test and training partition of size len(raw_df)*(1-test_split) and len(raw_df)*test_split respectively
            These sets are appended to the train_sets, test_sets array
            Returns index of partitioned sets in train_sets, test_sets

        '''
        if dataset_index == None: dataset_index = len(self.train_sets)

        test_ss = ShuffleSplit(n_splits=1, test_size=test_split)

        # Train and Test are disjoint partitions
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

        if debug_level > 1:
            print_df_details(self.train_sets[dataset_index]['rating'],'test_df')
            print_df_details(self.test_sets[dataset_index]['rating'],'train_df')

        return dataset_index

    def record_analysis_metrics(self, split, k, kfold_mse, kfold_mae, train_mse, train_mae, test_mse, test_mae, dataset_index=None):

        if dataset_index == None: dataset_index = len(self.analysis_sets)

        self.analysis_sets.insert(
            dataset_index,
            self.analysis_dict(
                split=split,
                k=k,
                kfold_mse=kfold_mse,
                kfold_mae=kfold_mae,
                train_mse=train_mse,
                train_mae=train_mae,
                test_mse=test_mse,
                test_mae=test_mae
            )
        )

    def data_dict(self, rating=[], rated=[], index=[]):
        return { 'rating': rating, 'rated': rated, 'size': npy.array(rating).shape, 'index': index }

    def analysis_dict(self, split, k, kfold_mse, kfold_mae, train_mse, train_mae, test_mse, test_mae):
        return {
            'split': split, 'k': k,
            'kfold_mse': kfold_mse, 'kfold_mae': kfold_mae,
            'train_mse': train_mse, 'train_mae': train_mae,
            'test_mse': test_mse, 'test_mae': test_mae
            }

def get_kNN(k, neighbours_df, feature_vector, distance_function, *args):



    return kNN_index

def euclidean_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("In euclidean_distance: Length of vector v1 and v2 do not match\n", "len(v1): ", len(v1), '\tlen(v2): ', len(v2))
    print((((v1-v2)**2).sum())**(0.5))
    # return (((v1-v2)**2).sum())**(0.5)

def kfold_validation_generator(data, kfolds=6, randomise=True):
    """
        Takes data and generates k training, validation pairs
        training and validation sets are disjoint sets of lengths len(data)/k (training) and (k-1)*len(data)/k (validation)
    """

    kfold_split = KFold(n_splits=kfolds, shuffle=randomise)

    for train_index, validation_index in kfold_split.split(data):
        yield train_index, validation_index
        # print(train_index, validation_index)


def print_metrics(analysis_dict, title, raw_results=False):
    print  title,
    print "Analysis Results "
    print analysis_dict['split']
    print "kFold k =", analysis_dict['k']
    # kfold training metrics (averaged)
    print "Kfold Training Set"
    if raw_results:
        print "Raw MSE Results:\t", analysis_dict['train_mse']
        print "Raw MAE Results:\t", analysis_dict['train_mae']
    print "Mean MSE:\t", sum(analysis_dict['train_mse']) / analysis_dict['k']
    print "Mean MAE:\t", sum(analysis_dict['train_mae']) / analysis_dict['k']
    # kfold testing metrics (averaged)
    print "Kfold Validation Set"
    if raw_results:
        print "Raw MSE Results:\t", analysis_dict['kfold_mse']
        print "Raw MAE Results:\t", analysis_dict['kfold_mae']
    print "Mean MSE:\t", sum(analysis_dict['kfold_mse']) / analysis_dict['k']
    print "Mean MAE:\t", sum(analysis_dict['kfold_mae']) / analysis_dict['k']
    # Test Holdout set metrics
    print "Test Holdout Set"
    print "Test MSE:\t", analysis_dict['test_mse']
    print "Test MAE:\t", analysis_dict['test_mae']


def print_df_details(df, name):
    print('***', name, '***', 'Shape: ', df.shape)

def cost_function(w, num_users, num_movies, num_features, train_df):
    X = (w[:num_movies*num_features]).reshape(num_movies, num_features)
    Theta = (w[num_movies*num_features:]).reshape(num_users, num_features)
    J = (((X.dot(Theta.T)-train_df['rating'])*train_df['rated'])**2).sum()/2 + (Theta**2).sum()*lamb/2 + (X**2).sum()*lamb/2
    if debug_level > 1:
        print("J = ", J)
    return J

def grad_function(w, num_users, num_movies, num_features, train_df):
    X = (w[:num_movies*num_features]).reshape(num_movies, num_features)
    Theta = (w[num_movies*num_features:]).reshape(num_users, num_features)
    X_grad = ((X.dot(Theta.T)-train_df['rating'])*train_df['rated']).dot(Theta) + X*lamb
    Theta_grad = (((X.dot(Theta.T)-train_df['rating'])*train_df['rated']).T).dot(X) + Theta*lamb
    X_grad = X_grad.reshape(-1)
    Theta_grad = Theta_grad.reshape(-1)
    return npy.append(X_grad, Theta_grad)

def mean_square_error(v1, v2):
    return ((v1-v2)**2).sum() /v1.size

def mean_absolute_error(v1,v2):
    return abs(v1-v2).sum() / v1.size

def nearest_rating(x):
    return npy.array([0 if i<0 else 5 if i>5 else round(i*2)/2 for i in x])

def train_model(num_users, num_movies, num_features, train_df):
    X = npy.random.rand(num_movies, num_features).reshape(-1)
    Theta = npy.random.rand(num_users, num_features).reshape(-1)

    res = minimize(cost_function, npy.append(X, Theta), args=(num_users, num_movies, num_features, train_df) , method='CG', jac = grad_function, options={'maxiter': 150})

    wopt = res.x
    X = (wopt[:num_movies*num_features]).reshape(num_movies, num_features)
    Theta = (wopt[num_movies*num_features:]).reshape(num_users, num_features)
    return X, Theta


def test_prediction(X, Theta, df):
    hypo0 = (X.dot(Theta.T)*df['rated']).reshape(-1)
    rated1 = df['rated'].reshape(-1)
    hypo = nearest_rating(hypo0)

    prediction = npy.array([j for i,j in enumerate(hypo) if rated1[i]])
    truth = npy.array([i for i in df['rating'].reshape(-1) if i>0])

    if debug_level > 1:
        print("Prediction is : ", prediction)
        print("Truth : ", truth)

    return mean_square_error(prediction, truth), mean_absolute_error(prediction, truth)



def main():

    d = dataset(path='./ml-latest-small/', dataset='ratings')
    index = d.random_sample_split()

    if debug_level:
        print(d.train_sets[index]['size'])

    kfold_mse = []
    kfold_mae = []
    ktrain_mse = []
    ktrain_mae = []
    fold = 0

    t_v_generator = kfold_validation_generator(d.train_sets[index]['index'], kfolds=6)

    for i, j in t_v_generator:
        fold += 1

        train_index = d.train_sets[index]['index'][i]
        validation_index = d.train_sets[index]['index'][j]

        # then do something with this fold
        train_df = d.df_from_index_list(index_list=train_index, df_size=d.master_df['size'])

        X, Theta = train_model(
                        num_users=d.master_df['size'][1],
                        num_movies=d.master_df['size'][0],
                        num_features=num_features,
                        train_df=train_df
                    )

        if debug_level > 1:
            print("X is : ", X)
            print("Theta is : ", Theta)
        # print("Cost min = ", cost_function(npy.append(X.reshape(-1), Theta.reshape(-1))))

        train_mse, train_mae = test_prediction(X, Theta, train_df)

        ktrain_mse.insert(len(ktrain_mse), train_mse)
        ktrain_mae.insert(len(ktrain_mae), train_mae)


        validation_df = d.df_from_index_list(index_list=validation_index, df_size=d.master_df['size'])

        val_mse, val_mae = test_prediction(X, Theta, validation_df)

        kfold_mse.insert(len(kfold_mse), kfold_mse)
        kfold_mae.insert(len(kfold_mse), kfold_mae)


        input("Press Enter to continue...")

    X, Theta = train_model(
                    num_users=d.master_df['size'][1],
                    num_movies=d.master_df['size'][0],
                    num_features=num_features,
                    train_df=d.train_sets[index]
                )

    test_mse, test_mae = test_prediction(X, Theta, d.test_sets[index])

    d.record_analysis_metrics(
        split="random_sample_split",
        k=len(kfold_mse),
        kfold_mse=kfold_mse,
        kfold_mae=kfold_mae,
        train_mse=ktrain_mse,
        train_mae=ktrain_mae,
        test_mse=test_mse,
        test_mae=test_mae
    )


    print_metrics(d.analysis_sets[index], "kfold mse + mae + holdout test set")

''' you will need to change variable names below to match '''
# with open('./records_full.txt', 'a') as f:
#     s = "\n"+str(num_features)+"\t"+str(lamb)+"\t"+str(partial)+"\t"+str(train_error)+"\t"+str(cv_error)+"\t"+str(mse)
#     f.write(s)
    # print("Final error rate : ", abs(hypo.sum()-tru.sum())/tru.sum())

    # npy.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    # print("Cost min = ", cost_function(npy.append(X.reshape(-1), Theta.reshape(-1))))

if __name__ == '__main__':
    import sys
    main()

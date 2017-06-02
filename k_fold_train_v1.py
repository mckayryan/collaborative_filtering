
import random
import numpy as npy
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize
# from train_X_Theta_v1 import *

random.seed()

num_movies = 0
num_users = 0
num_features = 100
training_rated = []
training_rating = []
lamb = 10

class dataset(object):

    def __init__(self, path, dataset):
        self.raw_df = pd.DataFrame(pd.read_csv(path + '%s.csv' % dataset))
        '''
            create index values for each unique value of movieId
        '''
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

        for _, data_index in enumerate(index_list):
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

def cost_function(w):
    X = (w[:num_movies*num_features]).reshape(num_movies, num_features)
    Theta = (w[num_movies*num_features:]).reshape(num_users, num_features)
    J = (((X.dot(Theta.T)-training_rating)*training_rated)**2).sum()/2 + (Theta**2).sum()*lamb/2 + (X**2).sum()*lamb/2
    print("J = ", J)
    return J

def grad_function(w):
    X = (w[:num_movies*num_features]).reshape(num_movies, num_features)
    Theta = (w[num_movies*num_features:]).reshape(num_users, num_features)
    X_grad = ((X.dot(Theta.T)-training_rating)*training_rated).dot(Theta) + X*lamb
    Theta_grad = (((X.dot(Theta.T)-training_rating)*training_rated).T).dot(X) + Theta*lamb
    X_grad = X_grad.reshape(-1)
    Theta_grad = Theta_grad.reshape(-1)
    return npy.append(X_grad, Theta_grad)

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
    global num_features
    num_features = 100

    global num_movies
    num_movies = d.master_df['size'][0]
    X = npy.random.rand(num_movies, num_features).reshape(-1)

    global num_users
    num_users = d.master_df['size'][1]
    Theta = npy.random.rand(num_users, num_features).reshape(-1)

    w0 = npy.append(X, Theta)

    t_v_generator = kfold_validation_generator(d.train_sets[index]['index'], kfolds=6)
    for i, j in t_v_generator:
        train_index = d.train_sets[index]['index'][i]
        validation_index = d.train_sets[index]['index'][j]
        # then do something with this fold
        train_df = d.df_from_index_list(index_list=train_index, df_size=d.master_df['size'])
        global training_rating
        training_rating = train_df['rating']
        global training_rated
        training_rated = train_df['rated']

        validation_df = d.df_from_index_list(index_list=validation_index, df_size=d.master_df['size'])
        validation_rating = validation_df['rating']
        validation_rated = validation_df['rated']

        res = minimize(cost_function, w0, method='CG', jac = grad_function, options={'maxiter': 300})

        wopt = res.x
        X = (wopt[:num_movies*num_features]).reshape(num_movies, num_features)
        Theta = (wopt[num_movies*num_features:]).reshape(num_users, num_features)
        print("X is : ", X)
        print("Theta is : ", Theta)
        print("Cost min = ", cost_function(npy.append(X.reshape(-1), Theta.reshape(-1))))
        hypo = (X.dot(Theta.T)*validation_rated).reshape(-1)
        tru = validation_rating.reshape(-1)
        print("Error rate : ", abs(hypo.sum()-tru.sum())/tru.sum())
        w0 = res.x
        input("Press Enter to continue...")

    test_rating = d.test_sets[index]['rating']
    test_rated = d.test_sets[index]['rated']
    hypo = (X.dot(Theta.T)*test_rated).reshape(-1)
    tru = test_rating.reshape(-1)
    print("Final error rate : ", abs(hypo.sum()-tru.sum())/tru.sum())

    npy.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
    print("Cost min = ", cost_function(npy.append(X.reshape(-1), Theta.reshape(-1))))

if __name__ == '__main__':
    import sys
    main()

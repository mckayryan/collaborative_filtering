# coding: utf-8

import numpy as npy
import pandas as pd
#from cost_plus_grad import cost_plus_grad
from numpy import linalg as lla
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize


# must change this path
movies_data = pd.read_csv('./ml-latest-small/%s.csv' % 'ratings')
movies_df = pd.DataFrame(movies_data)

# in order to translate movieId into movie_index
temp_df = movies_df.sort_values(['movieId'])
(m, n) = temp_df.shape
last = temp_df['movieId'].iloc[0]
movie_index = 0
movie_dic = {last : movie_index}
for i in range(1, m):
	if last != temp_df['movieId'].iloc[i]:
		last = temp_df['movieId'].iloc[i]
		movie_index = movie_index+1
		movie_dic[last] = movie_index

rating_matrix = movies_df.pivot(index='movieId', columns='userId', values='rating').as_matrix()
rated_matrix = ~npy.isnan(rating_matrix)
rating_matrix = npy.nan_to_num(rating_matrix)

temp = pd.DataFrame(movies_data[::100])
for i in range(1, 8):
	temp = temp.append(pd.DataFrame(movies_data[i::100]))
training_df = temp
training_rating = npy.zeros(rating_matrix.shape)
training_rated = npy.zeros(rating_matrix.shape, dtype = bool)

validation_df = pd.DataFrame(movies_data[8::100])
validation_rating = npy.zeros(rating_matrix.shape)
validation_rated = npy.zeros(rating_matrix.shape, dtype = bool)

test_df = pd.DataFrame(movies_data[9::100])
test_rating = npy.zeros(rating_matrix.shape)
test_rated = npy.zeros(rating_matrix.shape, dtype = bool)

# training_rating representation : num_movies * num_users

# training_matrix
(m, n) = training_df.shape
for i in range(m):
	mv = training_df['movieId'].iloc[i]
	us = training_df['userId'].iloc[i]
	rt = training_df['rating'].iloc[i]
	training_rating[movie_dic[mv], us-1] = rt
	training_rated[movie_dic[mv], us-1] = True

# validation_matrix
(m, n) = validation_df.shape
for i in range(m):
	mv = validation_df['movieId'].iloc[i]
	us = validation_df['userId'].iloc[i]
	rt = validation_df['rating'].iloc[i]
	validation_rating[movie_dic[mv], us-1] = rt
	validation_rated[movie_dic[mv], us-1] = True

# test_matrix
(m, n) = test_df.shape
for i in range(m):
	mv = test_df['movieId'].iloc[i]
	us = test_df['userId'].iloc[i]
	rt = test_df['rating'].iloc[i]
	test_rating[movie_dic[mv], us-1] = rt
	test_rated[movie_dic[mv], us-1] = True

[num_movies, num_users] = training_rating.shape
num_features = 200

# num_users = 671
# num_movies = 9066

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

def train_X_Theta(nf = 200, lb = 10):
	X = npy.random.rand(num_movies, nf)
	X = X.reshape(-1)
	Theta = npy.random.rand(num_users, nf)
	Theta = Theta.reshape(-1)
	w0 = npy.append(X, Theta)
	global num_features
	num_features = nf
	global lamb
	lamb = lb
	# wopt = fmin_bfgs(cost_function, w0, fprime=grad_function)
	res = minimize(cost_function, w0, method='CG', jac = grad_function, options={'maxiter': 300})
	# res = minimize(cost_function, w0, method='BFGS', jac = grad_function)
	# res = minimize(cost_function, w0, method='Newton-CG', jac = grad_function)
	wopt = res.x
	X = (wopt[:num_movies*num_features]).reshape(num_movies, num_features)
	Theta = (wopt[num_movies*num_features:]).reshape(num_users, num_features)
	return X, Theta


# print(J)
# print(X_grad)
# print(Theta_grad)
# print(training_rating)
# print(training_rated)

if __name__ == '__main__':
	import sys
	try:
		num_features = int(sys.argv[1])
	except IndexError:
		num_features = 200
		print("No feature number set. Default set to 200.")
	try:
		lamb = int(sys.argv[2])
	except IndexError:
		lamb = 10
		print("No lambda set. Default set to 10.")

	X, Theta = train_X_Theta(num_features, lamb)
	npy.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
	print("X is : ", X)
	print("Theta is : ", Theta)
	print("Cost min = ", cost_function(npy.append(X.reshape(-1), Theta.reshape(-1))))
	hypo = (X.dot(Theta.T)*validation_rated).reshape(-1)
	tru = validation_rating.reshape(-1)
	print("Hypothesis : ", hypo[npy.nonzero(hypo)])
	print("Truth : ", tru[npy.nonzero(tru)])

	print("Error rate : ", abs(hypo.sum()-tru.sum())/tru.sum())

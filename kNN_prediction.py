from k_fold_train_v4 import *

def test_kNN(k_values, neighbours_df, test_df, train_df):

    pcs_mse = npy.empty(0)
    pcs_mae = npy.empty(0)
    cs_mse = npy.empty(0)
    cs_mae = npy.empty(0)


    for k in k_values:

        pcs_predict, cs_predict, truth = predict_kNN(k, neighbours_df, test_df, train_df)

        pcs_mse = npy.insert(pcs_mse, len(pcs_mse), mean_square_error(pcs_predict, truth))
        cs_mse = npy.insert(cs_mse, len(cs_mse), mean_square_error(cs_predict, truth))



    plot_title = "kNN Users MSE Results PSC vs CS"
    plot_curve( title=plot_title,
                ylim=None,
                xlab="k Value", ylab="Mean Squared Error",
                ploty_1=pcs_mse, plotx_1=k_values, plot_lab1="Pearson Correlation Similarity (PCS)",
                ploty_2=cs_mse, plotx_2=k_values, plot_lab2="Cosine Similarity (CS)",
                fillrange_1=0, fillrange_2=0
    )

    file_name = plot_title+".png"
    plot.savefig(file_name, bbox_inches='tight')

def predict_kNN(k, neighbours_df, test_df, train_df):

    test_movies, test_users = npy.where( test_df['rated'] == True)

    kNN_pcs_prediction = npy.empty(0)
    kNN_cs_prediction = npy.empty(0)
    truth = npy.empty(0)

    no_ratings = {}

    for i in range(len(test_movies)):

        feature_index = test_movies[i]
        user_index = test_users[i]


        rated_index = npy.where( train_df['rated'][feature_index,:] == True)

        truth = npy.insert(truth, len(truth), test_df['rating'][feature_index, user_index])

        if len(rated_index[0]) > 0:

            kNN_pcs = get_kNN(k, neighbours_df, rated_index[0].tolist(), user_index, pearsons_correlation_similarity)
            kNN_cs = get_kNN(k, neighbours_df, rated_index[0].tolist(), user_index, cosine_similarity)

            kNN_pcs_prediction = npy.insert(kNN_pcs_prediction, len(kNN_pcs_prediction), npy.mean([train_df['rating'][feature_index, user] for user in kNN_pcs] ))
            kNN_cs_prediction = npy.insert(kNN_cs_prediction, len(kNN_cs_prediction), npy.mean( [train_df['rating'][feature_index, user] for user in kNN_cs] ))

        else:


            kNN_pcs_prediction = npy.insert(kNN_pcs_prediction, len(kNN_pcs_prediction), npy.mean([value for value in train_df['rating'][:, user_index] if value > 0] ))
            kNN_cs_prediction = npy.insert(kNN_cs_prediction, len(kNN_cs_prediction), npy.mean([value for value in train_df['rating'][:, user_index] if value > 0] ))

            no_ratings[feature_index] = 1


    return kNN_pcs_prediction, kNN_cs_prediction, truth

def get_kNN(k, neighbours_matrix, rated_index, feature_index, distance_function):

    rated_neighbours = { index: neighbours_matrix[index] for index in rated_index }

    user_distances = [ [i, distance_function(neighbours_matrix[feature_index], neighbours_matrix[i])] for i in rated_neighbours.keys() ]

    sorted_user_distances = sorted(user_distances, key=ig(1) )

    return [ user for user, _ in sorted_user_distances[-k:] ]


def euclidean_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("In euclidean_distance: Length of vector v1 and v2 do not match\n", "len(v1): ", len(v1), '\tlen(v2): ', len(v2))
    return npy.sum( npy.square( npy.subtract(v1,v2) ) )**(0.5)

def pearsons_correlation_similarity(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("In pearsons_correlation_similarity: Length of vector v1 and v2 do not match\n", "len(v1): ", len(v1), '\tlen(v2): ', len(v2))
    mean_v1 = npy.mean(v1)
    mean_v2 = npy.mean(v2)
    stdev_v1 = ( npy.sum( npy.square( v1 - mean_v1 ) ) )**(0.5)
    stdev_v2 = ( npy.sum( npy.square( v2 - mean_v2 ) ) )**(0.5)
    cov_v1_v2 = npy.dot(v1 - mean_v1, v2 - mean_v2 )
    return cov_v1_v2 / (stdev_v1*stdev_v2 + 0.0000000001)

def cosine_similarity(v1, v2):
    return npy.dot(v1,v2) / ( npy.dot(v1,v1)**(0.5) * npy.dot(v2,v2)**(0.5) )

# -*- coding: utf-8 -*-
"""
gwr-tb :: Growing Episodic Memory demo
@last-modified: 20 November 2018
@author: German I. Parisi (german.parisi@gmail.com)

"""

import gtls
from episodic_gwr import EpisodicGWR

if __name__ == "__main__":

    # Import dataset from file
    data_flag = True
    # Import pickled network from fileflag=
    import_flag = False
    # Train AGWR with imported dataset    
    train_flag = True
    train_type = 0 # 0:Batch, 1: Incremental
    # Compute classification accuracy    
    test_flag = True
    # Export pickled network to file       
    export_flag = False    
    # Plot network (2D projection)
    plot_flag = True
    
    if data_flag:
        ds_iris = gtls.IrisDataset(file='iris.csv', normalize=True)
        print("%s from %s loaded." % (ds_iris.name, ds_iris.file))

    if import_flag:
        file_name = 'my_gwr.egwr'
        my_net = gtls.import_network(file_name, EpisodicGWR)

    if train_flag:
        
        assert train_type < 2, "Invalid type of training."
        
        ds_vectors = ds_iris.vectors
        ds_labels = ds_iris.labels

        epochs = 15 # epochs per sample for incremental learning
        a_threshold = 0.85
        beta = 0.7
        learning_rates = [0.2, 0.001]
        context = True
        regulated = False
        my_net = EpisodicGWR(ds=ds_iris, num_context=0)
        
        if train_type == 0:
            # Batch training
            my_net.train_egwr(ds_vectors, ds_labels, epochs, a_threshold, beta,
                             learning_rates, context, regulated)
        else:
            # Incremental training
            batch_size = 10 # number of samples per epoch
            for s in range(0, ds_vectors.shape[0], batch_size):
                my_net.train_egwr(ds_vectors[s:s+batch_size], ds_labels[s:s+batch_size],
                                  epochs, a_threshold, beta, learning_rates, context,
                                  regulated)
                                  
        if export_flag:
            file_name = 'my_gwr.egwr'
            gtls.export_network(file_name, my_net)

        if test_flag:
            my_net.test_gammagwr(ds_vectors, ds_labels, test_accuracy=True)
            print("Accuracy on test-set: %s" % my_net.test_accuracy)

    if plot_flag:
        gtls.plot_gamma(my_net, edges=True, labels=True)
        

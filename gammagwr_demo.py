# -*- coding: utf-8 -*-
"""
gwr-tb :: Gamma-GWR demo
@last-modified: 30 November 2018
@author: German I. Parisi (german.parisi@gmail.com)

"""

import gtls
from gammagwr import GammaGWR

if __name__ == "__main__":

    # Import dataset from file
    data_flag = True
    # Import pickled network
    import_flag = False
    # Train AGWR with imported dataset    
    train_flag = True
    # Compute classification accuracy    
    test_flag = True
    # Export pickled network     
    export_flag = False    
    # Plot network (2D projection)
    plot_flag = True
    
    if data_flag:
        ds_iris = gtls.IrisDataset(file='iris.csv', normalize=True)
        print("%s from %s loaded." % (ds_iris.name, ds_iris.file))

    if import_flag:
        fname = 'my_net.ggwr'
        my_net = gtls.import_network(fname, GammaGWR)

    if train_flag:
       # Create network 
       my_net = GammaGWR()
       # Initialize network with two neurons
       my_net.init_network(ds=ds_iris, random=False, num_context=1)
       # Train network on dataset
       my_net.train_ggwr(ds=ds_iris, epochs=15, a_threshold=0.85, beta=0.7, l_rates=[0.2, 0.001])
       
    if test_flag:
        my_net.test_gammagwr(ds_iris, test_accuracy=True)
        print("Accuracy on test-set: %s" % my_net.test_accuracy)
 
    if export_flag:
        fname = 'my_net.ggwr'
        gtls.export_network(fname, my_net)

    if plot_flag:
        gtls.plot_gamma(my_net, edges=True, labels=True)
        
# -*- coding: utf-8 -*-
"""
gwr-tb :: Associative GWR demo
@last-modified: 17 November 2018
@author: German I. Parisi (german.parisi@gmail.com)

"""

import gtls
from agwr import AssociativeGWR

if __name__ == "__main__":

    # Import dataset from file
    data_flag = True
    # Import pickled network from fileflag=
    import_flag = False
    # Train AGWR with imported dataset    
    train_flag = True
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
        file_name = 'my_agwr.agwr'
        my_net = gtls.import_network(file_name, AssociativeGWR)

    if train_flag:
       my_net = AssociativeGWR(ds=ds_iris, random=False)
       my_net.train_agwr(ds_iris, epochs=15, a_threshold=0.85, 
                         learning_rates=[0.2, 0.001])

    if test_flag:
        my_net.test_agwr(ds_iris)
        print("Accuracy on test-set: %s" % my_net.test_accuracy)
 
    if export_flag:
        file_name = 'my_agwr.agwr'
        gtls.export_network(file_name, my_net)       

    if plot_flag:
        gtls.plot_network(my_net, edges=True, labels=True)

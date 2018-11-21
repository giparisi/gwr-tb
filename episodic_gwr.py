# -*- coding: utf-8 -*-
"""
gwr-tb :: Episodic-GWR
@last-modified: 20 November 2018
@author: German I. Parisi (german.parisi@gmail.com)

"""

import numpy as np
import math
from gammagwr import GammaGWR

class EpisodicGWR(GammaGWR):
    
    def __init__(self, ds, **kwargs):

        if ds is not None:
            # Number of neurons
            self.num_nodes = 2
            # Dimensionality of weights
            self.dimension = ds.vectors.shape[1]
            # Start with two neurons with context
            self.num_context = kwargs.get('num_context', 0)
            self.depth = self.num_context + 1
            self.weights = np.zeros((self.num_nodes, self.depth, self.dimension))
            # Global context
            self.g_context = np.zeros((self.depth, self.dimension))         
            # Temporal connections
            self.temporal = np.zeros((self.num_nodes, self.num_nodes))               
            # Habituation counters
            self.habn = np.ones(self.num_nodes)
            # Connectivity matrix
            self.edges = np.ones((self.num_nodes, self.num_nodes))
            # Age matrix
            self.ages = np.zeros((self.num_nodes, self.num_nodes))
            # Label histogram
            self.num_classes = ds.num_classes
            self.alabels = np.zeros((self.num_nodes, ds.num_classes))
            
            init_ind = list(range(0, self.num_nodes))
            for i in range(0, len(init_ind)):
                self.weights[i, 0] = ds.vectors[i]
                self.alabels[i, int(ds.labels[i])] = 1
                
            # Context coefficients
            self.alphas = self.compute_alphas(self.depth)
            
        # Keep unlocked to train. Lock to prevent training.
        self.locked = False

    def update_temporal(self, current_ind, previous_ind, **kwargs):
        new_node = kwargs.get('new_node', False)
        if new_node:
            self.temporal.resize((self.num_nodes, self.num_nodes))
        if previous_ind != -1 and previous_ind != current_ind:
            self.temporal[previous_ind, current_ind] += 1

    def remove_isolated_nodes(self):
        ind_c = 0
        rem_c = 0
        while (ind_c < self.num_nodes):
            neighbours = np.nonzero(self.edges[ind_c])
            if len(neighbours[0]) < 1:
                self.weights = np.delete(self.weights, ind_c, axis=0)
                self.alabels = np.delete(self.alabels, ind_c, axis=0)
                self.edges = np.delete(self.edges, ind_c, axis=0)
                self.edges = np.delete(self.edges, ind_c, axis=1)
                self.ages = np.delete(self.ages, ind_c, axis=0)
                self.ages = np.delete(self.ages, ind_c, axis=1)
                self.temporal = np.delete(self.temporal, ind_c, axis=0)
                self.temporal = np.delete(self.temporal, ind_c, axis=1)
                self.habn = np.delete(self.habn, ind_c)
                self.num_nodes -= 1
                rem_c += 1
            else:
                ind_c += 1
        print ("(-- Removed %s neuron(s))" % rem_c)
         
    def train_egwr(self, ds_vectors, ds_labels, epochs, a_threshold, beta, 
                   learning_rates, context, regulated):
        
        assert not self.locked, "Network is locked. Unlock to train."
        
        self.samples = ds_vectors.shape[0]        
        self.max_epochs = epochs
        self.a_threshold = a_threshold   
        self.epsilon_b, self.epsilon_n = learning_rates
        self.beta = beta
        self.regulated = regulated
        self.context = context
        if not self.context:
            self.g_context.fill(0)
        self.hab_threshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = self.samples # OK for batch, bad for incremental
        self.max_neighbors = 6
        self.max_age = 6000
        self.new_node = 0.5
        self.a_inc = 1
        self.a_dec = 0.1
        self.mod_rate = 0.01
            
        # Start training
        error_counter = np.zeros(self.max_epochs)
        previous_bmu = np.zeros((self.depth, self.dimension))
        previous_ind = -1
        for epoch in range(0, self.max_epochs):
            for iteration in range(0, self.samples):
                
                # Generate input sample
                self.g_context[0] = ds_vectors[iteration]
                label = ds_labels[iteration]
                
                # Update global context
                for z in range(1, self.depth):
                    self.g_context[z] = (self.beta * previous_bmu[z]) + ((1-self.beta) * previous_bmu[z-1])
                
                # Find the best and second-best matching neurons
                b_index, b_distance, s_index = self.find_bmus(self.g_context, second_best=True)
                
                b_label = np.argmax(self.alabels[b_index, :])
                misclassified = b_label != label
                
                # Quantization error
                error_counter[epoch] += b_distance
                
                # Compute network activity
                a = math.exp(-b_distance)

                # Store BMU at time t for t+1
                previous_bmu = self.weights[b_index]

                if (not self.regulated) or (self.regulated and misclassified):
                    
                    if (a < self.a_threshold
                        and self.habn[b_index] < self.hab_threshold
                        and self.num_nodes < self.max_nodes):
                        # Add new neuron
                        n_index = self.num_nodes
                        super().add_node(b_index)
                       
                        # Add label histogram
                        super().update_labels(n_index, label, new_node=True)                   
    
                        # Update edges and ages
                        super().update_edges(b_index, s_index, new_index=n_index)
                        
                        # Update temporal connections
                        self.update_temporal(n_index, previous_ind, new_node=True)
    
                        # Habituation counter                    
                        super().habituate_node(n_index, self.tau_b, new_node=True)
                    
                    else:
    
                        # Habituate BMU
                        super().habituate_node(b_index, self.tau_b)
    
                        # Update BMU's weight vector
                        b_rate = self.epsilon_b
                        n_rate = self.epsilon_n
                        if self.regulated and misclassified:
                            b_rate *= self.mod_rate
                            n_rate *= self.mod_rate
                        else:
                            # Update BMU's label histogram
                            super().update_labels(b_index, label)
    
                        super().update_weight(b_index, b_rate)
    
                        # Update BMU's edges // Remove BMU's oldest ones
                        super().update_edges(b_index, s_index)
    
                        # Update temporal connections
                        self.update_temporal(b_index, previous_ind)
    
                        # Update BMU's neighbors
                        super().update_neighbors(b_index, n_rate)
                        
                previous_ind = b_index

            # Remove old edges
            super().remove_old_edges()
            
            # Average quantization error (AQE)
            error_counter[epoch] /= self.samples
            
            print ("(Epoch: %s, NN: %s, ATQE: %s)" % (epoch+1, self.num_nodes, error_counter[epoch]))
            
        # Remove isolated neurons
        self.remove_isolated_nodes()
        
        print("Network size: %s" % self.num_nodes)

    def test_gammagwr(self, ds_vectors, ds_labels, **kwargs):
        test_accuracy = kwargs.get('test_accuracy', None)
        test_samples = ds_vectors.shape[0]
        self.bmus_index = -np.ones(test_samples)
        self.bmus_label = -np.ones(test_samples)
        self.bmus_activation = np.zeros(test_samples)
        
        input_context = np.zeros((self.depth, self.dimension))
        
        if test_accuracy:
            acc_counter = 0
        
        for i in range(0, test_samples):
            input_context[0] = ds_vectors[i]
            # Find the BMU
            b_index, b_distance = super().find_bmus(input_context)
            self.bmus_index[i] = b_index
            self.bmus_activation[i] = math.exp(-b_distance)
            self.bmus_label[i] = np.argmax(self.alabels[b_index])
            
            for j in range(1, self.depth):
                input_context[j] = input_context[j-1]
            
            if test_accuracy:
                if self.bmus_label[i] == ds_labels[i]:
                    acc_counter += 1

        if test_accuracy:
            self.test_accuracy =  acc_counter / ds_vectors.shape[0]

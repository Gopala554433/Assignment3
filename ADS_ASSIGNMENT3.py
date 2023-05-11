# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:22:59 2023

@author: Gopal
"""

"""
required import modules for 
processing graphs for fitting
and clistering of data
 """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy.stats import linregress

#pandas data frame is used for read csv file
dataset = pd.read_csv("assig3.csv")


def clustering(data_frames):
    """
This function clusters a list of dataframes using k-means algorithm. The function first scales the data using standard scaling technique and then finds the optimum number of clusters using the elbow method. It then fits the k-means algorithm to the dataset and plots the clusters and their centroids on the original data. 
"""
    year = "2000"
    for df in data_frames:
        x = df.values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Finding the optimum number of clusters
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++',
                            max_iter=300, n_init=10, random_state=42)
            kmeans.fit(x_scaled)
            wcss.append(kmeans.inertia_)

        # Plotting the elbow graph
        plt.plot(range(1, 11), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

        # Fitting kmeans to the dataset
        kmeans = KMeans(n_clusters=4, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        y_kmeans = kmeans.fit_predict(x_scaled)

        # Plotting the clusters on original data
       df <- df[complete.cases(df), ]
       df$region <- as.factor(df$region)
       prepare_scatter_plot(df, "ln_gdp_capita", "ln_co2_emissions_capita", 
                     color = "region", size = "population", loess = 1)

        plt.title(f'Clusters of countries ' + year)
        plt.xlabel("in gdp_capita")
        plt.ylabel("in Co2_emmission_capita")

        plt.legend()
        plt.show()
        year = "2020"

clustering(dfs_cluster)


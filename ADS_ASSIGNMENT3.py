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

def fitting_predictions():
    '''
    This function takes in a dataset and performs curve fitting and linear regression to predict CO2 emissions from liquid fuel consumption (kt) based on Urban population growth in selected countries. It also generates a plot to show the best fitting function, confidence range, and plots to show the predicted values in 10 and 20 years. 
    '''
    

    # Create a DataFrame with selected countries
    data_frame_fit = pd.DataFrame({
        "emission control": two_pop_fit,
        "co2_con_capita": two_fo_fit})
    data_frame_fit = data_frame_fit.dropna(axis=0)
    
    # Get the x and y data
    xdata = data_frame_fit.iloc[:, 0].values
    ydata = data_frame_fit.iloc[:, 1].values

    # Define the function to fit
    def func(x, a, b):
        return a*x + b

    # Fit the data
    popt, pcov = curve_fit(func, xdata, ydata)

    # Generate a plot showing the best fitting function
    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, 'o', label='data')
    ax.plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    ax.set_xlabel('Xv')
    ax.set_ylabel('Yv)')
    ax.legend()
    plt.show()

    # Compute the confidence ranges
    def err_ranges(popt, pcov):
        perr = np.sqrt(np.diag(pcov))
        lower = popt - perr
        upper = popt + perr
        return lower, upper

    # Generate a plot showing the confidence range
    
 # Use the model for predictions
    slope, intercept, r_value, p_value, std_err = linregress(xdata, ydata)

    # Predict the values in 10 years
    xpred = 10
    ypred = slope * xpred + intercept
    

    # Predict the values in 20 years
    xpred = 20
    ypred = slope * xpred + intercept
    


fitting_predictions()


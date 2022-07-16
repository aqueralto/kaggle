import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, ShuffleSplit, learning_curve
from src import utils


# Function to create different barcharts
def plot_barcharts(df: pd.DataFrame, x_label: str, y_label: str, hue: str, colors: list, cols: int, figsize: tuple):
    '''Plot a barchart for each category in the dataframe.

        Args:
            df (pd.DataFrame): Dataframe to plot.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            hue (str): Label for the hue.
            colors (list): List of colors to use for the bars.
            cols (int): Number of columns to use.
            figsize (tuple): Figure size.

        Returns:
            plot object
    '''
    
    # Create the figure
    _, axs = plt.subplots(ncols = cols, figsize = figsize)

    if cols == 2:
        # Countplot
        sns.countplot(x = x_label, data = df, hue=hue, ax = axs[0], palette = colors)

        if hue is None:
            axs[0].set_title('Countplot of ' + x_label, fontsize=18)
        else:
            axs[0].set_title('Countplot of ' + x_label + ' by ' + hue, fontsize=18)

        axs[0].set_ylabel('Counts', fontsize=16)

        # Barplot
        sns.barplot(x = x_label, y = y_label, data = df, hue=hue, ax = axs[1], palette = colors)
        
        if hue is None:
            axs[1].set_title('Barplot of ' + x_label + ' vs ' + y_label, fontsize=18)
        else:
            axs[1].set_title('Barplot of ' + x_label + ' vs ' + y_label + ' by ' + hue, fontsize=18)

        axs[1].set_ylabel(y_label, fontsize=16)
    
    elif cols == 3:
        # Countplot
        sns.countplot(x = x_label, data = df, hue=hue, ax = axs[0], palette = colors)
        
        if hue is None:
            axs[0].set_title('Countplot of ' + x_label, fontsize=18)
        else:
            axs[0].set_title('Countplot of ' + x_label + ' by ' + hue, fontsize=18)

        axs[0].set_ylabel('Counts', fontsize=16)

        # Countplot per category
        sns.countplot(x=x_label, hue=hue, data=df, ax=axs[1], palette=colors[:-1])

        axs[1].set_title('Passengers per ' + x_label + ' and ' + hue, fontsize=18)
        axs[1].set_ylabel('Counts', fontsize=16)

        # Barplot
        sns.boxplot(x = x_label, y = y_label, data = df, hue=hue, ax = axs[2], palette = colors)
        
        if hue is None:
            axs[2].set_title('Barplot of ' + x_label + ' vs ' + y_label, fontsize=18)
        else:
            axs[2].set_title('Barplot of ' + x_label + ' vs ' + y_label + ' by ' + hue, fontsize=18)
        
        axs[2].set_ylabel(y_label, fontsize=16)

    for i in range(cols):
        axs[i].tick_params(labelsize=16)
        axs[i].set_xlabel(x_label, fontsize=16)

    plt.tight_layout()
    plt.show()


# Define correlation function for one variable
def correlation_single_variable(df: pd.DataFrame, var: str) -> plt:
    """
    This function takes a dataframe and a variable name as input and returns a correlation plot for the variable.
    """
    # Create a correlation matrix for a single variable
    var_corr = pd.DataFrame(df.corr()[var])

    # Create a plot
    fig, ax = plt.subplots(figsize=(40, 4))
    sns.heatmap(var_corr.T, annot = True, square=True, annot_kws={"size": 25}, cmap='viridis')

    # Set the title
    ax.set_title('Correlation plot for ' + var, fontsize=40)

    # Customize the plot
    plt.tick_params(axis='both', which='major', labelsize=30)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)

    # Show the plot
    plt.show()

    # Return the correlation df
    return var_corr


# Function to plot histograms
def plot_histograms(df_ohe: pd.DataFrame, df_original: pd.DataFrame, x_label: str, hue: list, colors: list, figsize: tuple, bins: int):
    """
    Function that loops through a list of variables in a dataframe and plots a histogram for each variable.
    """

    if len(hue) == 1:
        # Create the figure
        _, axs = plt.subplots(nrows = 1, ncols = 1, figsize = figsize)

        # Plot the histogram
        sns.histplot(data=df_ohe, x=x_label, hue=df_original[hue[0]], color=colors[0], label=hue[0], bins=bins, ax=axs[0])

        # Set the title
        axs[0].set_title('Histogram of ' + x_label + ' by ' + hue[0], fontsize=18)

        # Set the labels
        axs[0].set_xlabel(x_label, fontsize=16)
        axs[0].set_ylabel('Counts', fontsize=16)

    elif len(hue) % 2 == 0 and len(hue) < 7:
        print(len(hue))
        # Create figure
        _, axs = plt.subplots(nrows=2, ncols=len(hue)//2, figsize=figsize)

        # Loop through the columns
        for i in range(len(hue)):
            if i < len(hue)//2:
                # Plot the histogram
                sns.histplot(data=df_ohe, x=x_label, hue=df_original[hue[i]], label=hue[i], bins=bins, ax=axs[0, i])
            elif i >= len(hue)//2 and i < len(hue):
                # Plot the histogram
                sns.histplot(data=df_ohe, x=x_label, hue=df_original[hue[i]], label=hue[i], bins=bins, ax=axs[1, i-(len(hue)//2)])

    # Customize the plot
    for ax in axs.flat:
        ax.set(xlabel=x_label, ylabel='Count')
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

        # Set y-axis scale
        ax.set_yscale('function', functions=(utils.forward, utils.inverse))

        # Customize the legends
        legend = ax.get_legend()

        # Get title and set fontsize
        title = legend.get_title()
        title.set_fontsize(20)

        # Get text and set fontsize
        text = legend.get_texts()

        for t in text:
            t.set_fontsize(18)
    
    plt.tight_layout()
    plt.show()


# Function to plot model accuracies
def plot_model_accuracies(df: pd.DataFrame):
    """
    This function takes a dataframe as input and plots the model accuracies.
    """
    # Create figure
    plt.figure(figsize=(10,5))

    # Define boxplot
    sns.boxplot(data=df, x='accuracy', y='model', showfliers=False)

    # Define swarmplot
    sns.swarmplot(data=df, x='accuracy', y='model', size=5, linewidth=1)

    # Customize the plot
    plt.title('Model Accuracies', fontsize=20)
    plt.xlabel('Accuracy', fontsize=18)
    plt.ylabel('Model', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.show()


# Function to plot learning curves
def learning_curve_plotter(estimator, X, y, ax=None, cv=None, n_jobs=None,
                   train_sizes=np.linspace(.1, 1.0, 5), legend=True):
    """ Plots the test and training learning curve of an estimator.

    Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Args:
        estimator (sklearn estimator): An estimator instance implementing `fit`
            and `predict` methods which will be cloned for each validation.
        X (array-like of shape (n_samples, n_features)): Training vector, where
            `n_samples` is the number of samples and `n_features` is the number
            of features.
        y (array-like of shape (n_samples) or (n_samples, n_features)): Target
            relative to ``X`` for classification or regression.
        ax (axes object, optional): Axes to plot learning curve.
        cv (cross-valiation generator, optional): Defaults to a `ShuffleSplit`
            with 100 splits for a smooth curve.
        train_sizes (array-like, optional): Points on training curve.
        legend (boolean, optional): Whether to plot a legend.

    Returns:
        axes object

    """

    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    if cv is None:
        cv = ShuffleSplit(n_splits=100, test_size=0.2)

    ax.set_xlabel('Training examples', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Calculate learning curve.
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    ax.grid()
    ax.fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1,
        color='darkorchid'
    )
    ax.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1,
        color='sandybrown'
    )
    ax.plot(train_sizes, train_scores_mean, 'o-', color='darkorchid', label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='sandybrown', label='CV score')
    
    if legend:
        ax.legend(loc='best')

    return ax

# Function used to plot the learning curves from a list of models
def plot_learning_curves(models: list, X_train: pd.DataFrame, y_train: pd.Series, 
                        fig_size: tuple, cv_plot: int, cv_lc: int):
    """ Plots the test and training learning curve of an estimator.

    Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """

    # Create figure
    _, axes = plt.subplots(nrows=len(models), ncols=2, figsize=fig_size, sharey='row', sharex='all')

    # Flatten the axes array
    axes = axes.flatten()

    # Create an empty dataframe to store the results
    df_results = pd.DataFrame(index=range(cv_plot))

    # Iterate over each one of the models
    for i, classifier in enumerate(models):
        # Load the completed grid search from disk.
        classifier_name = classifier.__class__.__name__
        grid_search = joblib.load(
            os.path.join(f'models/grid_search_{classifier_name}.pkl')
        )
        # Evaluate both the base model and the best model from the grid search.
        for j, (model, name) in enumerate([
            (classifier, classifier_name + '_base'), 
            (grid_search.best_estimator_, classifier_name)]):

            # Plot the learning curve for the model.
            ax = axes[i*2+j]
            learning_curve_plotter(
                model, X_train, y_train, cv=cv_lc, ax=ax, n_jobs=-1, legend=(i == 0 and j == 1)
            )
            ax.set_title(name, fontsize=18)

            # Store the accuracy of the model.
            df_results[name] = cross_val_score(
                model, X_train, y_train, scoring='accuracy', cv=cv_plot, n_jobs=-1
            )

    # Create a dataframe with the results
    df_results = pd.melt(df_results, var_name='model', value_name='accuracy')

    # Return the dataframe with the results
    return df_results
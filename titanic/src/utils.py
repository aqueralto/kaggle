import os
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.metrics import accuracy_score
# from src import kaggle_api

# Function x**(1/2) for Y-axis scale
# Source: https://matplotlib.org/stable/gallery/scales/scales.html#sphx-glr-gallery-scales-scales-py
def forward(x):
    return x**(1/2)

def inverse(x):
    return x**2


# Function to assign the value of the Family Group column based on the value of sibsp_parch column
def fam_vals(sum_sibsp_parch):
    """
    Function to assign the value of the Family Group column based on
    the value of the sum of the sibsp and parch columns
    """
    if sum_sibsp_parch == 1:
        return 'A'
    elif sum_sibsp_parch > 1 and sum_sibsp_parch <= 3:
        return 'S'
    elif sum_sibsp_parch > 3 and sum_sibsp_parch <= 6:
        return 'M'
    elif sum_sibsp_parch > 6:
        return 'L'


# Function used to create a dataframe with encoded categorical variables using OneHotEncoder
def encode_categorical_variables(df: pd.DataFrame, cat_vars: list) -> pd.DataFrame:
    """Function that encodes the categorical variables of a dataframe using OneHotEncoder,
       concatenates the original dataframe with the encoded on, and drops the original
       columns that were encoded.

       Returns: the original dataframe with the encoded columns.
    
    """
    # Encode the categorical variables
    for var in cat_vars:
        # Store name of variables
        cat_var_name = df[var].unique()

        # Sort the variables
        cat_var_name.sort()

        # Create column names for the encodings
        cat_var_name_encoded = [var + '_' + str(i) for i in cat_var_name]

        # Instantiate the encoder
        encoder = OneHotEncoder()

        # Fit the encoder and save the encoded variables into a dataframe
        encoded_vars = encoder.fit_transform(df[var].values.reshape(-1, 1)).toarray()
        encoded_vars_df = pd.DataFrame(encoded_vars, columns=cat_var_name_encoded)

        # Set the index of the encoded dataframe to the index of the original dataframe
        encoded_vars_df.index = df.index

        # Concatenate the encoded variables to the original dataframe and save it as a new dataframe
        df = pd.concat([df, encoded_vars_df], axis=1)

    # Drop the original categorical variables
    df.drop(cat_vars, axis=1, inplace=True)

    return df


# Function to create the list of variables for imputation
def create_var_list(df_corr: pd.DataFrame(), var_to_impute: str, threshold: float) -> list:
    """
    Function to create the list of variables for imputation based on the correlation
    coefficients and a threshold value.
    """

    # Calculate the absolute values of the correlation matrix
    df_corr_abs = df_corr.abs()

    # Sort the values in descending order
    df_corr_abs.sort_values(by=var_to_impute, ascending=False).T

    # Filter the values below a certain threshold
    df_corr_abs_filtered = df_corr_abs[df_corr_abs[var_to_impute] > threshold]

    # Drop the column 'var_to_impute' from the dataframe
    df_corr_abs_filtered.drop([var_to_impute], axis=0, inplace=True)

    # Create a list with the variables used for the imputation
    X_vars = df_corr_abs_filtered.index.tolist()

    # Return
    return X_vars


# Function to impute data on a specific column using KNNImputer
def knn_imputation(df: pd.DataFrame(), X_vars: list, var_to_impute: str, k: int) -> pd.DataFrame():
    """
    Function to impute data on a specific column using KNNImputer and a 
    selection of important variables create using the *create_var_list* function.
    """
    
    X = df[X_vars + [var_to_impute]]
    imputer = KNNImputer(n_neighbors=k, copy=True)
    imputer.fit_transform(X)

    # Temporal df to store the imputed values
    temp_df = pd.DataFrame(imputer.transform(X), columns=X.columns)

    # Update the original dataframe column with the imputed values
    df[var_to_impute] = temp_df[var_to_impute]

    return df


# Create function to do K-fold Stratified cross validation on all models
def k_fold_stratified_cv(models: list, X: pd.DataFrame, y: pd.Series, k: int, random_state: int) -> pd.DataFrame:
    """
    Function to do K-fold Stratified cross validation on all models and return
    a dataframe with the scores of each model.
    """

    # Get the number of folds
    n_folds = k

    # Initialize the kfolds object
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Empty dataframe to store the results
    accuracies = pd.DataFrame(index=range(n_folds))

    # Loop through each model
    for model in models:

        # Calculate the accuracy for each fold
        accuracies[model.__class__.__name__] = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

    # Melt the dataframe to make it easier to plot
    accuracies = pd.melt(accuracies, var_name='model', value_name='accuracy')

    # Return the dataframe
    return accuracies


# Function for hyperparameter tuning
def hyperparameter_tuning(models: list, X_train: pd.DataFrame, y_train: pd.Series, param_grids: dict, cv: int) -> object:
    """
    Function to do hyperparameter tuning on all models and return 
    the best model, saving it in a file.
    """
    
    # Iterate each classifier
    for clf, param_grid in zip(models, param_grids):
        
        # Skip if a parameter search has already been run.
        save_path = os.path.join(
                                    f'models/grid_search_{clf.__class__.__name__}.pkl'
                                )
        if os.path.exists(save_path):
            continue
        
        # Run a parameter grid search for each model
        grid_search = GridSearchCV(clf, param_grid, cv=cv, verbose=True, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Save the grid search to disk
        return joblib.dump(grid_search, save_path)


# Function to display the best hyperparameters for each model
def display_best_params(classifiers: list) -> None:
    # Show the best parameters from the grid search for each classifier
    for classifier in classifiers:
        # Get the name of the classifier
        classifier_name = classifier.__class__.__name__

        # Load the classifier
        grid_search = joblib.load(f'models/grid_search_{classifier_name}.pkl')

        # Print the best parameters
        print(classifier_name, 'best parameters:')
        display(grid_search.best_params_)

# Function to do leave-one-out cross validation on all models
def leave_one_out_cv(classifiers: list, X_train: pd.DataFrame, y_train: pd.Series, param_grids: dict) -> list:
    """
    Function to do Leave-One-Out hyperparameter tuning on all models and return 
    the best model, saving it in a file.
    """
    # Define the leave-one-out cross validation object
    loo = LeaveOneOut()

    # Create a list to store the models
    models = []
    # Iterate each classifier
    for clf, param_grid in zip(classifiers, param_grids):

        # Get classifier name
        classifier_name = clf.__class__.__name__
        
        # Skip if a parameter search has already been run.
        save_path = os.path.join(
                                    f'models/LOO_{clf.__class__.__name__}.pkl'
                                )
        if os.path.exists(save_path):
            continue
        
        # Run a parameter grid search for each model
        loo_cv = GridSearchCV(clf, param_grid, cv=loo, verbose=True, n_jobs=-1)
        loo_cv.fit(X_train, y_train)

        # Get the best model
        models.append((classifier_name, loo_cv.best_estimator_))

        # Save the grid search to disk
        return joblib.dump(loo_cv, save_path), models

# Function to test the best model
def model_submission(models: list, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):

    # Load the file data/gender_submission.csv
    gender_submission = pd.read_csv('data/gender_submission.csv', header=0)

    #

    # Iterate on the models
    for model in models:

        # Get the name of the model
        model_name = model.__class__.__name__
            
        # Load the model
        grid_search = joblib.load(
            os.path.join(f'models/grid_search_{model_name}.pkl')
        )

        # Get the best estimator
        grid_search.best_estimator_.fit(X_train, y_train)

        # Predict the test set
        y_pred = grid_search.predict(X_test)

        # Get the accuracy of the model
        accuracy = accuracy_score(gender_submission['Survived'], y_pred)

        # Print accuracy
        print(f'Accuracy of {model_name} vs optimal submission is {accuracy}')

        # Create a submission dataframe
        submission = pd.DataFrame({'PassengerId': (X_test.index+1), 'Survived': y_pred})

        # Ensure that the values from the survival column are integers
        submission['Survived'] = submission['Survived'].astype(int)

        # Save the submission to disk
        submission.to_csv(f'submissions/{model_name}.csv', index=False)


# Function to test the best model with LOO
def model_submission_loo(models: list, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame):

    # Load the file data/gender_submission.csv
    gender_submission = pd.read_csv('data/gender_submission.csv', header=0)

    # Iterate on the models
    for model in models:

        # Get the name of the model
        model_name = model.__class__.__name__
            
        # Load the model
        loo = joblib.load(
            os.path.join(f'models/loo_{model_name}.pkl')
        )

        # Get the best estimator
        loo.best_estimator_.fit(X_train, y_train)

        # Predict the test set
        y_pred = loo.predict(X_test)

        # Get the accuracy of the model
        accuracy = accuracy_score(gender_submission['Survived'], y_pred)

        # Print accuracy
        print(f'Accuracy of {model_name} vs optimal submission is {accuracy}')

        # Create a submission dataframe
        submission = pd.DataFrame({'PassengerId': (X_test.index+1), 'Survived': y_pred})

        # Ensure that the values from the survival column are integers
        submission['Survived'] = submission['Survived'].astype(int)

        # Save the submission to disk
        submission.to_csv(f'submissions/loo_{model_name}.csv', index=False)
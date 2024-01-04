import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_feature_importance(model):

    importance_df = pd.DataFrame()
    importance_df["Feature"] = model.feature_name()
    importance_df["Importance"] = model.feature_importance()

    # Sort features by importance
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    importance_plot = sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Feature Importance Plot")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    return importance_plot


def plot_correlation_matrix(dataset):
    # Assuming 'dataset' is a pandas DataFrame
    correlation_matrix = dataset.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Create a seaborn heatmap with a coolwarm color palette
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)

    # Show the plot
    plt.show()


def plot_average_price(df, target):
    """
    Plots the average price of a property for each category of a categorical variable.

    Args:
        df: The dataset.
        target: The target variable.

    Returns:
        None.
    """

    # Select categorical variables with less than 10 categories.
    categorical_variables = [
        var for var in df.columns if df[var].nunique() < 10
    ]

    for var in categorical_variables:
        # Calculate the average price of a property for each category.
        averages = df.groupby(var)[target].mean().sort_values()

        # Create the bar chart with improved aesthetics.
        plt.figure(figsize=(10, 6))
        sns.barplot(x=averages.index, y=averages.values, palette="viridis")
        plt.title(f"Average property price by {var}")
        plt.xlabel(var)
        plt.ylabel(f"Average {target}")
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.grid(alpha=0.3)
        plt.show()

# Example usage:
# plot_average_price(your_dataframe, 'your_target_variable')
        

def plot_explained_variance(dataframe, target):
    # Extract features and target variable
    features = dataframe.drop(columns=[target])
    target_variable = dataframe[target]

    # Standardize the features
    features_standardized = StandardScaler().fit_transform(features)

    # Apply PCA
    pca = PCA()
    principal_components = pca.fit_transform(features_standardized)

    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Cumulative explained variance
    cumulative_explained_variance = explained_variance_ratio.cumsum()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
    plt.ylim(0, 1)
    plt.xlim(1)
    plt.title('Explained Variance Ratio')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()

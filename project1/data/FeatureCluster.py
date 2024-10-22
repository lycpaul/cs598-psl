import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

import pandas as pd
import numpy as np


class FeatureClusterer:
    def __init__(self, threshold: float = 0.9):
        """
        Initializes the FeatureClusterer with a specified correlation threshold.

        Parameters:
        - threshold: float
            The correlation threshold to apply when forming clusters.
        """
        self.threshold = threshold
        self.clusters = {}
        self.aggregated_feature_names = []

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fits the feature clustering on the provided DataFrame.

        Parameters:
        - df: pd.DataFrame
            The input dataframe with numerical features.
        """
        # Compute the correlation matrix
        corr_matrix = df.corr().abs()

        # Convert correlation to distance
        distance_matrix = 1 - corr_matrix

        # Perform hierarchical clustering
        self.linked = sch.linkage(distance_matrix, method='average')

        # Apply Agglomerative Clustering
        cluster = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.threshold,
            affinity='precomputed',
            linkage='average'
        )
        cluster.fit(distance_matrix)

        # Create a mapping from cluster labels to features
        for feature, label in zip(corr_matrix.columns, cluster.labels_):
            self.clusters.setdefault(label, []).append(feature)

        # Generate aggregated feature names
        self.aggregated_feature_names = []
        for cluster_features in self.clusters.values():
            if len(cluster_features) > 1:
                self.aggregated_feature_names.append(
                    f'cluster_{cluster_features[0]}')
            else:
                self.aggregated_feature_names.append(cluster_features[0])

        print(f"Identified {
              len(self.aggregated_feature_names)} feature clusters.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by aggregating features based on the fitted clusters.

        Parameters:
        - df: pd.DataFrame
            The input dataframe with numerical features.

        Returns:
        - pd.DataFrame
            The dataframe with clustered (aggregated) features.
        """
        aggregated_features = pd.DataFrame(index=df.index)
        for cluster_label, cluster_features in self.clusters.items():
            if len(cluster_features) > 1:
                # Aggregate by mean for clusters with multiple features
                aggregated_features[f'cluster_{
                    cluster_features[0]}'] = df[cluster_features].mean(axis=1)
            else:
                # Retain the feature as is if it's the only one in its cluster
                aggregated_features[cluster_features[0]
                                    ] = df[cluster_features[0]]

        print(f"Reduced features from {df.shape[1]} to {
              aggregated_features.shape[1]} by clustering.")
        return aggregated_features

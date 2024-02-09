import pandas as pd
import numpy as np

from collections import Counter
from nltk.tokenize import word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


def filter_top_labels(df, col, n):
    print(f"Top {n} {col}: {df[col].value_counts().nlargest(n).index.tolist()}")
    return df[df.isin(df[col].value_counts().nlargest(n).index)[col] == True]


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, target=None):
        self.features = features
        self.target = target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop rows with missing target values, select specified features and target, and drop duplicates
        X_selected = X.dropna(subset=self.target)[self.features].drop_duplicates(
            subset=self.features
        )
        return X_selected


class OutlierToNaN(BaseEstimator, TransformerMixin):
    def __init__(self, iqr=False):
        self.iqr = iqr

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.iqr:
            q1, q3 = np.percentile(X, [25, 75])
            lower_bound = q1 - 1.5 * (q3 - q1)
            upper_bound = q3 + 1.5 * (q3 - q1)
            X_copy[X < lower_bound] = np.nan
            X_copy[X > upper_bound] = np.nan
        elif str(X.dtype).startswith("float"):
            X_copy[X > 100] = np.nan
        return X_copy


class ContinuousImputer(BaseEstimator, TransformerMixin):
    def __init__(self, dist, scaler, n_miss, *prms):
        self.dist = dist
        self.scaler = scaler
        self.n_miss = n_miss
        self.params = [p for p in prms]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing_indices = X.index[X.isna()]
        new_data = self.dist.rvs(*self.params, size=self.n_miss)
        reverse_transform = self.scaler.inverse_transform(
            new_data.reshape(-1, 1)
        ).flatten()
        imputed_series = X.copy()
        imputed_series[missing_indices] = reverse_transform
        return imputed_series


class CatImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_cat):
        self.n_cat = n_cat

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Impute missing values using a random variable
        def random_imputing(row, choices, probs):
            if pd.isnull(row):
                return np.random.choice(choices, p=probs)
            else:
                return row

        # Calculate value counts of non-null values
        vc = X.dropna().value_counts()

        # Map top categories, treat others as 'other', and calculate normalized value counts
        top = X.map(
            {
                i: i if i in vc[: self.n_cat].index.tolist() else "misc"
                for i in vc.index.tolist()
            }
        )
        top_vc = top.value_counts(normalize=True)

        # Extract choices and corresponding probabilities
        self.choices = top_vc.index.tolist()
        self.probs = pd.to_numeric(top_vc.values, errors="coerce")

        # Apply the imputation function to the Series
        X_imputed = self.top.apply(
            random_imputing,
            args=(
                self.choices,
                self.probs,
            ),
        )
        return X_imputed


class WordEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, rank=20):
        self.rank = rank

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series):
        vectorizer = CountVectorizer(max_features=self.rank, binary=True)
        X_vectorized = vectorizer.fit_transform(X)
        print
        binary_vectors = pd.DataFrame(
            X_vectorized, columns=[f"{X.name}_word{i}" for i in range(self.rank)]
        )
        return binary_vectors


class LabelFormatter(BaseEstimator, TransformerMixin):

    def __init__(self, to_format=None):
        self.to_format = to_format

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Filter and extract nouns from a given sentence.
        def filter_tokens(sentence):
            if pd.notna(sentence):
                tokens = word_tokenize(sentence.lower())
                words = {
                    word
                    for word in tokens
                    # Extract nouns that do not contain specific characters
                    if all(char not in word for char in "%'/:") and len(word) > 2
                }
                return words
            else:
                return np.nan

        # Apply the filter_tokens function to extract and filter nouns from sentences
        if isinstance(X, pd.DataFrame):
            formatted_col = X[self.to_format].apply(filter_tokens)
            X.loc[:, self.to_format] = formatted_col
        else:
            X = pd.DataFrame(X.apply(filter_tokens))

        return X


class MultiLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, to_encode=None, rank=None):
        self.to_encode = to_encode
        self.rank = rank

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Function to filter words in a list based on a checklist
        def filter_words(lst, checklist):
            if pd.notna(lst):
                # Keep only words that are in the checklist
                words = {word for word in lst if word in checklist}
                return words
            else:
                return np.nan

        encoded_col = X[self.to_encode]
        # Count the occurrences of each word in the dataset
        count = Counter(
            [
                word
                for rows in encoded_col.dropna()
                for word in rows
                if word == word and "and" not in word
            ]
        )

        # Extract the most common words and their counts up to the specified rank
        self.most_common = pd.Series(
            [i[1] for i in count.most_common(self.rank)],
            index=[j[0] for j in count.most_common(self.rank)],
        )

        # Apply the filter_words function to keep only the most common words in each row
        encoded_col = encoded_col.apply(filter_words, args=(self.most_common.index,))

        # Initialize MultiLabelBinarizer and fit it with the most common words
        mlb = MultiLabelBinarizer()
        mlb.fit([self.most_common.index])

        # Transform the filtered words into binary representation
        encoded_values = mlb.transform(encoded_col.dropna()).tolist()

        # Apply the transformation to replace words with their binary representation
        encoded_col = encoded_col.apply(
            lambda x: (
                encoded_values.pop(0) if pd.notna(x) else np.full(self.rank, np.nan)
            )
        )

        X.loc[:, self.to_encode] = encoded_col
        return X


class LabelImputer(BaseEstimator, TransformerMixin):
    def __init__(self, to_impute=None):
        self.to_impute = to_impute

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        knn = KNNImputer(n_neighbors=10)
        imputed_col = knn.fit(X[self.to_impute])
        X.loc[:, self.target] = imputed_col
        return X

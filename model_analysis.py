from abc import ABC, abstractmethod
import os
import json
import sys
import random
from math import ceil

from icecream import ic  # type: ignore
from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score  # type: ignore
from xgboost import XGBClassifier  # type: ignore
from catboost import CatBoostClassifier, Pool  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.naive_bayes import BernoulliNB  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore
import shap  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.colors import TwoSlopeNorm  # type: ignore
import seaborn as sns  # type: ignore
from scipy.stats import wasserstein_distance  # type: ignore
from umap import UMAP  # type: ignore
import keras as K  # type: ignore
from scipy.stats import bootstrap  # type: ignore
from alibi.explainers import TreeShap  # type: ignore
import shap  # type: ignore


# ------------------------
# Load and preprocess data
# ------------------------


df_full = pd.read_csv(
    "data/bancocomvida_selecionado_30jan2024.csv",
    na_values="z.missing",
    sep=";",
    decimal=",",
)
df_full_unselected = pd.read_csv(
    "data/bancocomvidafull_30jan2024.csv",
    na_values="z.missing",
    sep=";",
    decimal=",",
)

df_full["e1_q56"] = df_full["e1_q56"].astype("category")
df_full["area_com"] = df_full["area_com"].astype("category")
df_full["income"] = df_full["income"].astype("category")
df_full["school"] = df_full["school"].astype("category")
df_full["bolsafamilia"] = df_full["bolsafamilia"].astype("category")
df_full["children"] = df_full["children"].astype("category")
df_full["adolescents"] = df_full["adolescents"].astype("category")
df_full["elders"] = df_full["elders"].astype("category")
df_full.info()

df_full["idade_calc"] = df_full_unselected["idade_calc"]
df_full["e1_q35"] = df_full_unselected["e1_q35"]
df_full["e1_q45"] = df_full_unselected["e1_q45"]
df_full["e1_q51"] = df_full_unselected["e1_q51"]
df_full["e1_q26_j"] = df_full_unselected["e1_q26_j"]
df_full["e1_q31"] = df_full_unselected["e1_q31"]
df_full["e1_q76"] = df_full_unselected["e1_q76"]

df_full = df_full.dropna(subset=["ebia3"])

df_full = df_full.rename(
    {c: c.replace(" ", "-").replace("<", "lt") for c in df_full.columns},
    axis=1,
)

categorical_features = [
    column_name
    for column_name in df_full.columns
    if df_full[column_name].dtype == "category"
]
for column_name in categorical_features:
    df_full[column_name] = df_full[column_name].cat.add_categories(["NA"]).fillna("NA")
    assert df_full[column_name].dtype == "category"
    assert not df_full[column_name].isna().all()


# -------------
# Define models
# -------------


class Model(ABC):
    @abstractmethod
    def train(self, X, Y, categorical_features):
        pass

    @abstractmethod
    def predict_proba1(self, X):
        pass


class LogisticRegressionModel(Model):
    def train(self, X, Y, categorical_features):
        self.categorical_features = categorical_features
        X_categorical = X[categorical_features]
        X_noncategorical = X.drop(categorical_features, axis=1)

        self.onehot = OneHotEncoder(handle_unknown="ignore").fit(X_categorical)

        X_categorical = self.onehot.transform(X_categorical).toarray()
        X_categorical = pd.DataFrame(
            X_categorical,
            columns=self.onehot.get_feature_names_out(),
        )
        X = X_noncategorical.join(X_categorical)
        X = X.fillna(0)

        self.model = LogisticRegression(random_state=0).fit(X, Y)

    def predict_proba1(self, X):
        X_categorical = X[self.categorical_features]
        X_noncategorical = X.drop(self.categorical_features, axis=1)
        X_categorical = self.onehot.transform(X_categorical).toarray()
        X_categorical = pd.DataFrame(
            X_categorical, columns=self.onehot.get_feature_names_out()
        )
        X = X_noncategorical.join(X_categorical)
        X = X.fillna(0)
        return self.model.predict_proba(X)[:, 1]


class DecisionTreeModel(Model):
    def train(self, X, Y, categorical_features):
        self.categorical_features = categorical_features
        X_categorical = X[categorical_features]
        X_noncategorical = X.drop(categorical_features, axis=1)

        self.onehot = OneHotEncoder(handle_unknown="ignore").fit(X_categorical)

        X_categorical = self.onehot.transform(X_categorical).toarray()
        X_categorical = pd.DataFrame(
            X_categorical,
            columns=self.onehot.get_feature_names_out(),
        )
        X = X_noncategorical.join(X_categorical)
        X = X.fillna(0)

        self.model = DecisionTreeClassifier(random_state=0).fit(X, Y)

    def predict_proba1(self, X):
        X_categorical = X[self.categorical_features]
        X_noncategorical = X.drop(self.categorical_features, axis=1)
        X_categorical = self.onehot.transform(X_categorical).toarray()
        X_categorical = pd.DataFrame(
            X_categorical, columns=self.onehot.get_feature_names_out()
        )
        X = X_noncategorical.join(X_categorical)
        X = X.fillna(0)
        return self.model.predict_proba(X)[:, 1]


class RandomForestModel(Model):
    def train(self, X, Y, categorical_features):
        self.categorical_features = categorical_features
        X_categorical = X[categorical_features]
        X_noncategorical = X.drop(categorical_features, axis=1)

        self.onehot = OneHotEncoder(handle_unknown="ignore").fit(X_categorical)

        X_categorical = self.onehot.transform(X_categorical).toarray()
        X_categorical = pd.DataFrame(
            X_categorical,
            columns=self.onehot.get_feature_names_out(),
        )
        X = X_noncategorical.join(X_categorical)
        X = X.fillna(0)

        self.model = RandomForestClassifier(random_state=0).fit(X, Y)

    def predict_proba1(self, X):
        X_categorical = X[self.categorical_features]
        X_noncategorical = X.drop(self.categorical_features, axis=1)
        X_categorical = self.onehot.transform(X_categorical).toarray()
        X_categorical = pd.DataFrame(
            X_categorical, columns=self.onehot.get_feature_names_out()
        )
        X = X_noncategorical.join(X_categorical)
        X = X.fillna(0)
        return self.model.predict_proba(X)[:, 1]


class NeuralNetworkModel(Model):
    def train(self, X, Y, categorical_features):
        self.categorical_features = categorical_features
        X_categorical = X[categorical_features]
        X_noncategorical = X.drop(categorical_features, axis=1)

        self.onehot = OneHotEncoder(handle_unknown="ignore").fit(X_categorical)

        X_categorical = self.onehot.transform(X_categorical).toarray()
        X_categorical = pd.DataFrame(
            X_categorical,
            columns=self.onehot.get_feature_names_out(),
        )
        X = X_noncategorical.join(X_categorical)
        X = X.fillna(0)

        self.model = K.Sequential(
            [
                K.layers.Dense(32, activation="gelu"),
                K.layers.Dense(16, activation="gelu"),
                K.layers.Dense(8, activation="gelu"),
                K.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(
            # optimizer="adam",  # E=53
            loss=K.losses.BinaryCrossentropy(),
            metrics=["accuracy", "auc"],
        )
        self.model.fit(
            X.to_numpy().astype("float32"),
            Y.to_numpy().astype("float32"),
            epochs=50,
            batch_size=32,
        )

    def predict_proba1(self, X):
        X_categorical = X[self.categorical_features]
        X_noncategorical = X.drop(self.categorical_features, axis=1)
        X_categorical = self.onehot.transform(X_categorical).toarray()
        X_categorical = pd.DataFrame(
            X_categorical, columns=self.onehot.get_feature_names_out()
        )
        X = X_noncategorical.join(X_categorical)
        X = X.fillna(0)
        return np.squeeze(self.model.predict(X.to_numpy().astype("float32")))


class XGBoostModel(Model):
    def train(self, X, Y, categorical_features):
        self.model = XGBClassifier(
            tree_method="hist", enable_categorical=True, random_state=0
        )
        self.model.fit(X, Y)

    def predict_proba1(self, X):
        return self.model.predict_proba(X)[:, 1]


class CatBoostModel(Model):
    def train(self, X, Y, categorical_features):
        self.model = CatBoostClassifier(
            random_state=0,
            cat_features=categorical_features,
        ).fit(X, Y)

    def predict_proba1(self, X):
        return self.model.predict_proba(X)[:, 1]


# -------------------------
# Train and Evaluate Models
# -------------------------


TARGET = "ebia3"
X_full = df_full.drop(TARGET, axis=1)
Y_full = df_full[TARGET]

X_train, X_test, Y_train, Y_test = train_test_split(
    X_full, Y_full, test_size=0.2, random_state=0
)

models = {
    "Logistic Regression": LogisticRegressionModel(),
    "Decision Tree": DecisionTreeModel(),
    "Random Forest": RandomForestModel(),
    "CatBoost": CatBoostModel(),
    "Neural Network": NeuralNetworkModel(),
    "XGBoost": XGBoostModel(),
}
for model_name, model in models.items():
    print(f"=== {model_name} === ")

    model.train(X_train, Y_train, categorical_features)

    pred_probas = model.predict_proba1(X_test)
    binary_preds = np.where(pred_probas >= 0.5, 1, 0)

    def btstp(func):
        result = bootstrap(
            (Y_test, pred_probas, binary_preds),
            func,
            paired=True,
        )
        center = (result.confidence_interval.high + result.confidence_interval.low) / 2
        radius = (result.confidence_interval.high - result.confidence_interval.low) / 2
        return f"{center:.2f} ± {radius:.2f}"

    print(
        f"  AUC ROC (test): {btstp(lambda Y_test_, pred_probas_, binary_preds_: roc_auc_score(Y_test_, pred_probas_))}"
    )
    print(
        f"  Accuracy (test): {btstp(lambda Y_test_, pred_probas_, binary_preds_: accuracy_score(Y_test_, binary_preds_))}"
    )
    print(
        f"  Precision (test): {btstp(lambda Y_test_, pred_probas_, binary_preds_: precision_score(Y_test_, binary_preds_))}"
    )
    print(
        f"  Recall (test): {btstp(lambda Y_test_, pred_probas_, binary_preds_: recall_score(Y_test_, binary_preds_))}"
    )

# best_model = models["XGBoost"]
# best_model = models["CatBoost"]
# best_model = models["Logistic Regression"]
# best_model = models["Decision Tree"]
# best_model = models["Random Forest"]
best_model = models["CatBoost"]
# best_model = models["Neural Network"]

# ------------------------
# Classical Explainability
# ------------------------

model: CatBoostClassifier = best_model.model  # type: ignore

feature_importances = model.get_feature_importance(
    Pool(X_test, Y_test, cat_features=categorical_features),
    type="LossFunctionChange",
)
feature_importance_order = np.argsort(feature_importances)[::-1]
print("Feature importance ranking:")
k = 0
most_important_features = []
feature_strengths = []
for i, j in enumerate(feature_importance_order):
    if feature_importances[j] >= 0.0005:
        most_important_features.append(X_full.columns[j])
        feature_strengths.append(feature_importances[j])
        print(f"  {i+1}. `{X_full.columns[j]}`  ({feature_importances[j]:.3f})")
        k += 1
print(f"  ... and {len(feature_importance_order) - k} other less relevant ones")

plt.barh(
    [
        {
            "idade_calc": "Age",  # "Age",
            "e1_q56": "Family income loss during the pandemic",  # "Impact of pandemic on income",
            "ppb": "Household density (PPB)",  # "People per Bedroom",
            "income": "Family income (nominal)",  # "Income",
            "area_com": "Favela/Community",  # "Region",
            "pessoasnacasa": "Household size",  # "People in home",
            "comodostotal": "Number of rooms in the household",  # "Number of rooms",
            "e1_q76": "Binge drinking",  # "Significant alcohol consumption",
            "e1_q31": "Health self-rating",  # "Self-rating of health",
            "bolsafamilia": "Bolsa Família benefit",  # "Bolsa-Família aid",
            "e1_q51": "Employment status",  # "Employed",
            "e1_q35": "Depressive symptoms",  # "Feeling depressed",
            "school": "Educational level",  # "School",  # XXX
            "e1_q17a": "Number of children in Bolsa Família",  # "N. of Children in Bolsa-Família aid",
            "bedroom": "Number of bedrooms in the household",  # "Number of bedrooms",
            "epidemicwk": "Epidemic week",  # "Epidemic wk",  # XXX
            "e1_q26_j": "History of depression diagnosis",  # "Diagnosed with depression",
            "e1_q45": "Impact of pandemic on life",
            "numberofpersonsage10upto17": "Adolescents in the household (yes/no)",  # "N. of Teenagers",
            "adolescents": "Number of Adolescents",  # "N. of Adolescents",
        }[name]
        for name in most_important_features[::-1]
    ],
    feature_strengths[::-1],
    alpha=0.8,
)
plt.xlabel("Reduction in log-loss by removing the feature")
plt.savefig("feature_importances.png", bbox_inches="tight")
plt.savefig("feature_importances.pdf", bbox_inches="tight")

# --------------------
# Conformal Prediction
# --------------------

ALPHA = 0.05


def conformity(X, Y):
    probs = best_model.predict_proba1(X)
    if isinstance(Y, int):
        if Y == 1:
            return 1 - probs
        else:
            return 1 - (1 - probs)
    else:
        return np.where(Y == 1, 1 - probs, 1 - (1 - probs))


def adjusted_quantile(xs, phi):
    return np.quantile(xs, phi * (1 + 1 / len(xs)))


scores = conformity(X_test, Y_test)
threshold_0 = adjusted_quantile(scores[Y_test == 0], 1 - ALPHA)
threshold_1 = adjusted_quantile(scores[Y_test == 1], 1 - ALPHA)
ic(threshold_0, threshold_1)


def btstp_mean(samples):
    result = bootstrap((samples,), np.mean)
    center = (result.confidence_interval.high + result.confidence_interval.low) / 2
    radius = (result.confidence_interval.high - result.confidence_interval.low) / 2
    return f"{center:.2f} ± {radius:.2f}"


confident_ia = conformity(X_test, 1) <= threshold_1
confident_noia = conformity(X_test, 0) <= threshold_0
print("Confident IA:", btstp_mean(confident_ia & ~confident_noia))
print("Confident no IA:", btstp_mean(~confident_ia & confident_noia))
print("Dunno:", btstp_mean(confident_ia & confident_noia))
print("Erroneous:", btstp_mean(~confident_ia & ~confident_noia))
print("Confident IA:", np.mean(confident_ia & ~confident_noia))
print("Confident no IA:", np.mean(~confident_ia & confident_noia))
print("Dunno:", np.mean(confident_ia & confident_noia))
print("Erroneous:", np.mean(~confident_ia & ~confident_noia))

# ----------------------
# Coverage plot via UMAP
# ----------------------

covers = (confident_ia & (Y_test == 1)) | (confident_noia & (Y_test == 0))
X_test = pd.get_dummies(X_test, columns=categorical_features)
X_test = X_test.fillna(0)  # XXXXXXXXX
X_test = StandardScaler().fit_transform(X_test)
# X_test_reduced = UMAP(verbose=True).fit_transform(X_test, y=covers)
X_test_reduced = UMAP(densmap=True, verbose=True, random_state=0).fit_transform(
    X_test,  # y=covers
)
ic(X_test_reduced)
ic(X_test_reduced.shape)

outlier_alpha = 0.01
min_x = np.quantile(X_test_reduced[:, 0], outlier_alpha) - 1
min_y = np.quantile(X_test_reduced[:, 1], outlier_alpha) - 1
max_x = np.quantile(X_test_reduced[:, 0], 1 - outlier_alpha) + 1
max_y = np.quantile(X_test_reduced[:, 1], 1 - outlier_alpha) + 1

plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)

filter = (
    (min_x <= X_test_reduced[:, 0])
    & (X_test_reduced[:, 0] <= max_x)
    & (min_y <= X_test_reduced[:, 1])
    & (X_test_reduced[:, 1] <= max_y)
)
plt.hexbin(
    X_test_reduced[filter, 0],
    X_test_reduced[filter, 1],
    C=covers[filter],
    gridsize=25,
    reduce_C_function=np.mean,
    alpha=0.8,
    cmap="viridis_r",
    # cmap="coolwarm_r",
    # norm=TwoSlopeNorm(vmin=0, vcenter=0.95, vmax=1),
)
cb = plt.colorbar()
cb.ax.axhline(0.95, c="b")

plt.axis("off")

plt.savefig("coverage_plot.png")
plt.savefig("coverage_plot.pdf")

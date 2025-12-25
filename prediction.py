import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve
)


def prediction(df):

    st.subheader("Prediction")

    # -------------------------------
    # Target selection
    # -------------------------------
    target = st.selectbox("Select target column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Encode features
    X = pd.get_dummies(X, drop_first=True)

    # -------------------------------
    # Problem type
    # -------------------------------
    problem_type = st.selectbox(
        "Prediction Type",
        ["Regression", "Classification"]
    )

    # ===============================
    # REGRESSION
    # ===============================
    if problem_type == "Regression":

        model_name = st.selectbox(
            "Select Regression Model",
            [
                "Linear Regression",
                "Ridge",
                "Lasso",
                "ElasticNet",
                "Decision Tree",
                "Random Forest"
            ]
        )

        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Ridge":
            model = Ridge()
        elif model_name == "Lasso":
            model = Lasso()
        elif model_name == "ElasticNet":
            model = ElasticNet()
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestRegressor(random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("R² Score:", r2_score(y_test, y_pred))
        st.write("MSE:", mean_squared_error(y_test, y_pred))

        # Predictions table
        results = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        })
        st.dataframe(results.head(10))

        # Actual vs Predicted
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color="red"
        )
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

    # ===============================
    # CLASSIFICATION
    # ===============================
    if problem_type == "Classification":

        # Block wrong target
        if y.dtype != "object" and y.nunique() > 10:
            st.error(
                "Selected target is continuous. "
                "Classification requires categorical target."
            )
            return

        # Encode target if needed
        if y.dtype == "object":
            y = y.astype("category").cat.codes

        model_name = st.selectbox(
            "Select Classification Model",
            [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest"
            ]
        )

        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.write("Accuracy:", accuracy_score(y_test, y_pred))

        # Confusion matrix + report
        col1, col2 = st.columns(2)

        with col1:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with col2:
            st.code(classification_report(y_test, y_pred))

        # Download predictions
        results = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred
        })

        st.download_button(
            "Download Predictions",
            results.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

        # ROC & PR curves (binary only)
        if len(y.unique()) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], "--")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend()
            st.pyplot(fig)

            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            fig, ax = plt.subplots()
            ax.plot(recall, precision)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            st.pyplot(fig)

        # Model comparison
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        scores = []
        for name, m in models.items():
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            scores.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds)
            })

        st.subheader("Model Comparison")
        st.dataframe(pd.DataFrame(scores))

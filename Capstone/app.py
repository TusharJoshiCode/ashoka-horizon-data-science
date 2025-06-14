import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ---- Page Config ----
st.set_page_config(page_title="Exoplanet Habitability", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: black;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 10px;
    }
    h1, h2, h3, h4, h5 {
        color: #f9f9f9;
    }
    .stMetric > div {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🪐 Exoplanet Habitability Prediction")
st.markdown("A Machine Learning Capstone Project by **Tushar Joshi**")

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("./data/phl_exoplanet_catalog_2019.csv")
    return df

df = load_data()
df = df.dropna(subset=["P_HABITABLE"])  # Remove rows with missing target
y = df["P_HABITABLE"]

# ---- Preprocessing ----
X_raw = df.select_dtypes(include=["float64", "int64"])
# Drop columns with all missing values
X_raw = X_raw.dropna(axis=1, how='all')
X_columns = X_raw.columns
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X_raw)
X_columns = X_raw.columns  # Now matches imputed data

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Sidebar Model ----
model_name = st.sidebar.selectbox("🔧 Choose Model", ["Logistic Regression", "Random Forest", "K-Nearest Neighbors"])
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "Random Forest":
    model = RandomForestClassifier()
else:
    model = KNeighborsClassifier()

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ---- Metrics and Report ----
col1, col2 = st.columns(2)
with col1:
    st.metric("✅ Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
with col2:
    st.write("### 📄 Classification Report")
    st.code(classification_report(y_test, y_pred), language="text")

# ---- Confusion Matrix ----
st.write("### 🔁 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm", 
            xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
st.pyplot(fig)

# ---- Feature Importances ----
if model_name == "Random Forest":
    st.write("### 🌟 Top Feature Importances")
    importances = pd.Series(model.feature_importances_, index=X_columns)
    top_feat = importances.sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots()
    top_feat.plot(kind='barh', ax=ax2, color='skyblue')
    st.pyplot(fig2)

# ---- Pie Chart of Class Distribution ----
st.write("### 🌍 Habitability Distribution")
label_counts = y.value_counts()
labels = label_counts.index.map(lambda val: "Habitable" if val else "Not Habitable")
fig3, ax3 = plt.subplots()
label_counts.plot.pie(autopct="%1.1f%%", labels=labels, colors=["#ff9999", "#66b3ff"], ax=ax3)
ax3.set_ylabel("")
st.pyplot(fig3)

# ---- Footer ----
st.markdown("---")
st.markdown("<center><i>Made with ❤️ by Tushar Joshi</i></center>", unsafe_allow_html=True)

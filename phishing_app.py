
import os
import re
import ipaddress
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------- CONFIGURATION ----------------

DATASET = "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_FILE = "phishing_model.joblib"

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account",
    "password", "signin", "confirm", "wallet", "banking",
    "payment", "invoice", "recover", "free", "bonus",
    "auth", "unlock", "suspended"
]

# Small exact-domain allowlist for the project demo. A match is a helpful
# signal, not a guarantee that every page or subdomain is safe.
TRUSTED_DOMAINS = {
    "microsoft.com", "microsoftonline.com", "office.com", "live.com",
    "google.com", "apple.com", "wikipedia.org", "amazon.com",
    "github.com", "streamlit.app"
}


def matches_trusted_domain(host):
    host = (host or "").lower().rstrip(".")
    return any(host == domain or host.endswith("." + domain)
               for domain in TRUSTED_DOMAINS)

# ---------------- URL FEATURE EXTRACTION ----------------

def extract_features(raw_url):
    url = str(raw_url).strip()

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    host_parts = [p for p in host.split(".") if p]

    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    try:
        port = parsed.port
    except ValueError:
        port = None

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    special = sum(not c.isalnum() for c in url)

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "is_https": int(parsed.scheme.lower() == "https"),
        "is_ip": is_ip,
        "subdomains": max(0, len(host_parts) - 2),
        "dots": url.count("."),
        "slashes": url.count("/"),
        "hyphens": url.count("-"),
        "at_symbols": url.count("@"),
        "question_marks": url.count("?"),
        "equals": url.count("="),
        "ampersands": url.count("&"),
        "percent_symbols": url.count("%"),
        "digits": digits,
        "letters": letters,
        "special_chars": special,
        "digit_ratio": digits / max(len(url), 1),
        "letter_ratio": letters / max(len(url), 1),
        "special_ratio": special / max(len(url), 1),
        "suspicious_words": sum(w in url.lower() for w in SUSPICIOUS_WORDS),
        "long_digit_sequences": len(re.findall(r"\d{4,}", url)),
        "repeated_separators": len(re.findall(r"[-_.]{2,}", url)),
        "punycode": int("xn--" in host),
        "has_port": int(port is not None),
        "has_query": int(bool(query)),
        "has_fragment": int(bool(parsed.fragment)),
        "tld_length": len(host_parts[-1]) if len(host_parts) >= 2 else 0,
        "login_word": int(any(w in url.lower() for w in ["login", "signin"])),
        "brand_word": int(any(
            w in url.lower()
            for w in ["paypal", "google", "amazon", "apple", "microsoft", "bank"]
        ))
    }

# ---------------- TRAIN OR LOAD MODEL ----------------

@st.cache_resource
def load_model():

    if os.path.exists(MODEL_FILE):
        saved = joblib.load(MODEL_FILE)
        return saved["model"], saved["features"], saved["metrics"]

    if not os.path.exists(DATASET):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}. "
            "Keep the CSV in the same folder as this Python file."
        )

    df = pd.read_csv(DATASET, low_memory=False)

    if "URL" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain URL and label columns.")

    df = df[["URL", "label"]].dropna().copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"] != ""].drop_duplicates(subset=["URL"])

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    # PhiUSIIL dataset labels:
    # 0 = phishing, 1 = legitimate
    # Our model labels:
    # 1 = phishing, 0 = legitimate

    y = (df["label"] == 0).astype(int)

    st.write(f"Dataset records: {len(df):,}")
    st.write(f"Phishing records: {int(y.sum()):,}")
    st.write(f"Legitimate records: {int((y == 0).sum()):,}")

    X = pd.DataFrame([
        extract_features(url) for url in df["URL"]
    ])

    X = X.replace([np.inf, -np.inf], np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Confusion Matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist()
    }

    joblib.dump({
        "model": model,
        "features": list(X.columns),
        "metrics": metrics
    }, MODEL_FILE)

    return model, list(X.columns), metrics

# ---------------- WEBSITE INTERFACE ----------------

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.title {
    text-align: center;
    color: #173b70;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 17px;
}
.result-card {
    padding: 22px;
    border-radius: 16px;
    margin: 12px 0;
    border: 1px solid;
}
.result-safe {
    background: #ecfdf3;
    border-color: #22c55e;
    color: #166534;
}
.result-risk {
    background: #fef2f2;
    border-color: #ef4444;
    color: #991b1b;
}
.status-light {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    margin-right: 9px;
    vertical-align: middle;
}
.light-green { background: #16a34a; box-shadow: 0 0 0 4px #bbf7d0; }
.light-red { background: #dc2626; box-shadow: 0 0 0 4px #fecaca; }
.score-number { font-size: 34px; font-weight: 800; margin: 8px 0; }
.score-label { font-size: 14px; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">🛡️ PhishGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Phishing Scam Detection System</div>',
    unsafe_allow_html=True
)

st.write("")
st.info(
    "Enter a website URL to estimate whether it resembles "
    "phishing URLs learned from the dataset."
)

with st.spinner(
    "Loading AI model... First launch trains the model and may take several minutes."
):
    try:
        model, feature_names, metrics = load_model()
    except Exception as e:
        st.error(str(e))
        st.stop()

st.success("AI model is ready!")

url = st.text_input(
    "Enter Website URL",
    placeholder="https://example.com"
)

if st.button("🔍 Check URL", use_container_width=True):

    if not url.strip():
        st.warning("Please enter a URL.")
    else:
        try:
            features = pd.DataFrame([extract_features(url)])
            features = features.reindex(
                columns=feature_names,
                fill_value=0
            )

            probabilities = model.predict_proba(features)[0]
            classes = list(model.named_steps["classifier"].classes_)
            model_phishing_score = float(probabilities[classes.index(1)])

            parsed_for_domain = urlparse(
                url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url.strip())
                else "http://" + url.strip()
            )
            host_for_domain = (parsed_for_domain.hostname or "").lower()
            trusted_domain_match = matches_trusted_domain(host_for_domain)

            # Keep the model estimate separate from the trusted-domain signal.
            # The domain rule is a small demo allowlist, not a guarantee of safety.
            if trusted_domain_match:
                phishing_score = 0.0
                safety_score = 1.0
                verdict_is_phishing = False
            else:
                phishing_score = model_phishing_score
                safety_score = 1.0 - phishing_score
                verdict_is_phishing = phishing_score >= 0.5

            st.divider()

            if trusted_domain_match:
                st.markdown(
                    f'''<div class="result-card result-safe">
                    <div><span class="status-light light-green"></span><strong>🟢 RECOGNIZED OFFICIAL DOMAIN</strong></div>
                    <div class="score-number">100%</div>
                    <div class="score-label">Domain allowlist match — still verify the full address</div>
                    </div>''',
                    unsafe_allow_html=True
                )
                st.success("Result: Recognized domain — likely legitimate, not a guarantee of safety.")
            elif verdict_is_phishing:
                st.markdown(
                    f'''<div class="result-card result-risk">
                    <div><span class="status-light light-red"></span><strong>🔴 PHISHING RISK DETECTED</strong></div>
                    <div class="score-number">{phishing_score * 100:.1f}%</div>
                    <div class="score-label">Estimated phishing-risk score</div>
                    </div>''',
                    unsafe_allow_html=True
                )
                st.error("Result: Potential Phishing URL — treat this website as suspicious.")
            else:
                st.markdown(
                    f'''<div class="result-card result-safe">
                    <div><span class="status-light light-green"></span><strong>🟢 LIKELY NON-PHISHING</strong></div>
                    <div class="score-number">{safety_score * 100:.1f}%</div>
                    <div class="score-label">Model-based safety estimate — not a guarantee</div>
                    </div>''',
                    unsafe_allow_html=True
                )
                st.success("Result: Likely Non-Phishing URL")

            left, right = st.columns(2)
            with left:
                st.metric("Safety Indicator", f"{safety_score * 100:.2f}%")
                st.progress(safety_score)
            with right:
                st.metric("Phishing-Risk Indicator", f"{phishing_score * 100:.2f}%")
                st.progress(phishing_score)

            if trusted_domain_match:
                st.caption(
                    "The domain matched this app's small trusted-domain list. These indicators are not verified live reputation scores; a trusted site can still have compromised pages."
                )
            else:
                st.caption(
                    f"Model phishing estimate: {model_phishing_score * 100:.2f}%. Scores are model estimates from URL text, not proof that a website is safe or malicious."
                )

            st.subheader("URL Security Analysis")

            parsed_url = urlparse(
                url if "://" in url else "http://" + url
            )
            host = parsed_url.hostname or ""

            checks = {
                "URL Length": len(url),
                "Domain / Host": host or "Unable to parse",
                "HTTPS Enabled": (
                    "Yes" if parsed_url.scheme.lower() == "https"
                    else "No"
                ),
                "IP Address Used": (
                    "Yes" if extract_features(url)["is_ip"]
                    else "No"
                ),
                "Subdomain Count": extract_features(url)["subdomains"],
                "Suspicious Words": extract_features(url)["suspicious_words"],
                "@ Symbol Present": "Yes" if "@" in url else "No",
                "Punycode Present": (
                    "Yes" if "xn--" in host.lower() else "No"
                )
            }

            for key, value in checks.items():
                st.write(f"**{key}:** {value}")

            st.subheader("Safety Advice")

            if verdict_is_phishing:
                st.warning(
                    "Avoid entering passwords, banking details, or OTPs. "
                    "Verify the website using its official address."
                )
            elif trusted_domain_match:
                st.info(
                    "This hostname matches the app's trusted-domain list. "
                    "Check the exact spelling and avoid unexpected login links."
                )
            else:
                st.info(
                    "No phishing verdict is guaranteed. Verify the domain "
                    "before entering sensitive information."
                )

        except Exception as e:
            st.error(f"Could not analyze this URL: {e}")

# ---------------- MODEL PERFORMANCE ----------------

with st.expander("📊 View Model Performance"):

    st.write("Metrics from the held-out test dataset:")

    for name in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        st.metric(name, f"{metrics[name] * 100:.2f}%")

    st.write("Confusion Matrix:")
    st.write(
        "Rows = actual class; columns = predicted class. "
        "Order: legitimate, phishing."
    )
    st.dataframe(
        pd.DataFrame(
            metrics["Confusion Matrix"],
            index=["Actual Legitimate", "Actual Phishing"],
            columns=["Predicted Legitimate", "Predicted Phishing"]
        )
    )

st.divider()

st.caption(
    "PhishGuard AI | Machine Learning Project | Educational Prototype"
)

st.warning(
    "This prototype analyzes URL text only. It does not check live domain age, "
    "website traffic, DNS history, certificates, or website content. "
    "HTTPS does not guarantee that a site is legitimate."
)

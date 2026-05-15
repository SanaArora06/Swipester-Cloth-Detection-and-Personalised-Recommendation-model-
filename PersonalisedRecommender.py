import numpy as np
import joblib
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
from pathlib import Path

from fashion_pipeline.utils import resize_image
from tensorflow.keras import Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.preprocessing import normalize


TARGET_SIZE = 224
PAD_COLOR = (255, 255, 255)
TOP_USER_COUNT = 5

st.set_page_config(page_title="Fashion Classifier", layout="wide")
st.title("Fashion Classifier")
st.caption(
    "This app classifies the uploaded clothing image and then ranks users who are most likely to be "
    "interested in it using swipe history, recency, and embedding similarity."
)


clf = joblib.load("demo_model.joblib")
le = joblib.load("label_encoder.joblib")


@st.cache_resource
def load_model():
    # Match the training-time EfficientNet backbone used to create the saved embeddings.
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(TARGET_SIZE, TARGET_SIZE, 3)),
    )
    x = GlobalAveragePooling2D()(base.output)
    return Model(base.input, x)


model = load_model()


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def scale_zero_one(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    if maximum - minimum < 1e-9:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - minimum) / (maximum - minimum)


def merge_category_for_preferences(value: str) -> str:
    normalized = str(value).strip().lower().replace(" / ", "_").replace(" ", "_")
    if normalized in {"tees_tanks", "graphic_tees", "shirts_polos", "blouses_shirts"}:
        return "tops"
    if normalized in {"jackets_coats", "jackets_vests"}:
        return "outerwear"
    if normalized in {"sweaters", "sweatshirts_hoodies", "cardigans"}:
        return "knitwear"
    if normalized in {"shorts", "skirts", "pants", "denim", "leggings", "suiting"}:
        return "bottoms"
    if normalized in {"rompers_jumpsuits", "rompers_/_jumpsuits"}:
        return "rompers_jumpsuits"
    return normalized


ITEM_LABEL_TO_CATEGORY = {
    "tees / tanks": "tops",
    "graphic tees": "tops",
    "shirts / polos": "tops",
    "blouses / shirts": "tops",
    "jackets / coats": "outerwear",
    "jackets / vests": "outerwear",
    "sweaters": "knitwear",
    "hoodies / sweatshirts": "knitwear",
    "cardigans": "knitwear",
    "shorts": "bottoms",
    "skirts": "bottoms",
    "pants": "bottoms",
    "denim": "bottoms",
    "leggings": "bottoms",
    "suiting": "bottoms",
    "dresses": "dresses",
    "rompers / jumpsuits": "rompers_jumpsuits",
}


@st.cache_data
def load_user_interest_data():
    joined_path = Path("outputs/metadata/swipster_phase3_joined.csv")
    summary_path = Path("outputs/metadata/swipster_user_category_interest.csv")
    profiles_path = Path("outputs/metadata/swipster_user_embedding_profiles.pkl")

    if joined_path.exists() and summary_path.exists() and profiles_path.exists():
        joined = pd.read_csv(joined_path).copy()
        if "timestamp" in joined.columns:
            joined["timestamp"] = pd.to_datetime(joined["timestamp"], errors="coerce")
        user_summary = pd.read_csv(summary_path).copy()
        if "last_timestamp" in user_summary.columns:
            user_summary["last_timestamp"] = pd.to_datetime(user_summary["last_timestamp"], errors="coerce")
        user_profiles = pd.read_pickle(profiles_path)
        if isinstance(user_profiles, dict):
            user_profiles = pd.DataFrame(
                [
                    {
                        "user_id": user_id,
                        "positive_swipe_count": value.get("positive_swipe_count", 0),
                        "profile_embedding": value.get("profile_embedding"),
                    }
                    for user_id, value in user_profiles.items()
                ]
            )
        elif not isinstance(user_profiles, pd.DataFrame):
            user_profiles = pd.DataFrame(user_profiles)

        if "user_id" not in user_profiles.columns:
            user_profiles = pd.DataFrame(
                columns=["user_id", "positive_swipe_count", "profile_embedding"]
            )
        if "positive_swipe_count" not in user_profiles.columns:
            user_profiles["positive_swipe_count"] = 0
        if "profile_embedding" not in user_profiles.columns:
            user_profiles["profile_embedding"] = None
        return joined, user_summary, user_profiles

    meta = pd.read_csv("outputs/metadata/embedding_metadata.csv").copy()
    emb = np.load("outputs/metadata/image_embeddings.npy")
    swipe_df = pd.read_csv("swipster_primarydata.csv").copy()

    meta["category_ml"] = meta["category"].apply(merge_category_for_preferences)
    swipe_df["image_filename"] = swipe_df["image_path"].apply(lambda path: Path(path).name)
    swipe_df["timestamp"] = pd.to_datetime(swipe_df["timestamp"], errors="coerce")
    swipe_df["response"] = swipe_df["response"].astype(str).str.strip().str.lower()
    swipe_df["response_binary"] = swipe_df["response"].map({"yes": 1, "no": 0}).fillna(0)
    swipe_df["response_time_sec"] = pd.to_numeric(swipe_df["response_time_sec"], errors="coerce")
    swipe_df["item_label_key"] = swipe_df["item_label"].astype(str).str.strip().str.lower()
    swipe_df["category_ml_from_label"] = swipe_df["item_label_key"].map(ITEM_LABEL_TO_CATEGORY)

    joined = swipe_df.merge(
        meta[["image_filename", "embedding_index", "category_ml"]],
        on="image_filename",
        how="left",
    )
    joined["category_ml"] = joined["category_ml"].fillna(joined["category_ml_from_label"])

    reference_time = joined["timestamp"].max()
    joined["days_since_event"] = (reference_time - joined["timestamp"]).dt.total_seconds().div(86400)
    joined["days_since_event"] = joined["days_since_event"].fillna(joined["days_since_event"].max())
    joined["recency_weight"] = np.exp(-joined["days_since_event"] / 30.0)

    median_response_time = joined["response_time_sec"].median()
    joined["response_time_sec"] = joined["response_time_sec"].fillna(median_response_time)
    joined["speed_weight"] = 1.0 / np.log1p(joined["response_time_sec"] + 1.0)
    joined["preference_signal"] = joined["response_binary"].map({1: 1.0, 0: -1.0})
    joined["interest_score"] = (
        joined["preference_signal"] * joined["recency_weight"] * joined["speed_weight"]
    )

    user_summary = (
        joined.groupby(["user_id", "category_ml"], dropna=False)
        .agg(
            interactions=("response", "size"),
            yes_count=("response_binary", "sum"),
            avg_response_time_sec=("response_time_sec", "mean"),
            avg_recency_weight=("recency_weight", "mean"),
            total_interest_score=("interest_score", "sum"),
            last_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )
    user_summary["yes_rate"] = user_summary["yes_count"] / user_summary["interactions"]

    positive = joined[
        (joined["response_binary"] == 1) & joined["embedding_index"].notna()
    ].copy()
    positive["embedding_index"] = positive["embedding_index"].astype(int)

    profiles = []
    for user_id, group in positive.groupby("user_id"):
        indices = group["embedding_index"].to_numpy()
        weights = group["recency_weight"].to_numpy()
        profile = normalize(np.average(emb[indices], axis=0, weights=weights).reshape(1, -1))[0]
        profiles.append(
            {
                "user_id": user_id,
                "positive_swipe_count": int(len(group)),
                "profile_embedding": profile,
            }
        )

    user_profiles = pd.DataFrame(profiles)
    if user_profiles.empty:
        user_profiles = pd.DataFrame(
            columns=["user_id", "positive_swipe_count", "profile_embedding"]
        )
    return joined, user_summary, user_profiles


def pil_to_embedding(image: Image.Image) -> np.ndarray:
    # Keep inference consistent with training: RGB -> padded/cropped 224x224 -> preprocess_input -> L2 normalize.
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(rgb, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    batch = preprocess_input(batch)
    emb = model.predict(batch, verbose=0)
    return normalize(emb)


def center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def estimate_foreground_crop(image: Image.Image) -> Image.Image:
    # Approximate the clothing/person region by detecting pixels that differ from the border background.
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    height, width, _ = rgb.shape

    border = np.concatenate(
        [
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.sqrt(((rgb - background) ** 2).sum(axis=2))

    threshold = max(25.0, float(np.percentile(distance, 82)))
    foreground = distance > threshold

    coords = np.argwhere(foreground)
    if coords.size == 0:
        return center_square_crop(image)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    box_h = y1 - y0 + 1
    box_w = x1 - x0 + 1
    if box_h < height * 0.2 or box_w < width * 0.2:
        return center_square_crop(image)

    pad_y = max(10, int(box_h * 0.08))
    pad_x = max(10, int(box_w * 0.08))

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)
    return image.crop((x0, y0, x1, y1))


def build_inference_views(image: Image.Image) -> dict[str, Image.Image]:
    rgb = ImageOps.exif_transpose(image).convert("RGB")

    # View 1: exact padded resize to mimic the training-time processed image layout.
    padded = resize_image(rgb, size=TARGET_SIZE, mode="pad", pad_color=PAD_COLOR)

    # View 2: centered square crop for images where the outfit occupies the middle of the frame.
    center_crop = resize_image(center_square_crop(rgb), size=TARGET_SIZE, mode="crop", pad_color=PAD_COLOR)

    # View 3: heuristic foreground crop for cluttered backgrounds or off-center subjects.
    foreground_crop = resize_image(estimate_foreground_crop(rgb), size=TARGET_SIZE, mode="pad", pad_color=PAD_COLOR)

    return {
        "padded_view": padded,
        "center_crop_view": center_crop,
        "foreground_crop_view": foreground_crop,
    }


def aggregate_scores(view_embeddings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack(view_embeddings)

    # LinearSVC exposes decision_function, not predict_proba, so average its class scores across views.
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(stacked)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        mean_scores = scores.mean(axis=0)
        pred_idx = int(np.argmax(mean_scores))
        return mean_scores, np.array([pred_idx], dtype=int)

    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(stacked)
        mean_probabilities = probabilities.mean(axis=0)
        pred_idx = int(np.argmax(mean_probabilities))
        return mean_probabilities, np.array([pred_idx], dtype=int)

    averaged_embedding = normalize(stacked.mean(axis=0, keepdims=True))
    pred = clf.predict(averaged_embedding)
    return np.array([], dtype=float), pred


def top_class_table(scores: np.ndarray) -> list[tuple[str, float]]:
    if scores.size == 0:
        return []

    ranking = np.argsort(scores)[::-1][:3]
    rows = []
    for idx in ranking:
        label = le.inverse_transform([idx])[0]
        rows.append((label, float(scores[idx])))
    return rows


def summarize_why_match(row: pd.Series, predicted_category: str) -> str:
    reasons = []
    if row["category_yes_rate"] >= 0.75:
        reasons.append(f"high yes-rate for {predicted_category}")
    elif row["category_yes_rate"] >= 0.5:
        reasons.append(f"positive history for {predicted_category}")

    if row["similarity_score"] >= 0.75:
        reasons.append("liked visually similar items")
    elif row["similarity_score"] >= 0.55:
        reasons.append("moderate embedding similarity")

    if row["recent_interest_score"] >= 0.55:
        reasons.append("recent activity in this category")

    if row["positive_swipe_count"] >= 10:
        reasons.append("strong interaction history")
    elif row["positive_swipe_count"] >= 5:
        reasons.append("repeat positive swipes")

    if not reasons:
        reasons.append("overall profile aligns with this item")
    return ", ".join(reasons[:3])


def match_users_to_item(item_embedding: np.ndarray, predicted_category: str) -> pd.DataFrame:
    _, user_summary, user_profiles = load_user_interest_data()

    category_rows = user_summary[user_summary["category_ml"] == predicted_category].copy()
    if category_rows.empty:
        category_rows = (
            user_summary.groupby("user_id", dropna=False)
            .agg(
                interactions=("interactions", "sum"),
                yes_count=("yes_count", "sum"),
                avg_response_time_sec=("avg_response_time_sec", "mean"),
                avg_recency_weight=("avg_recency_weight", "mean"),
                total_interest_score=("total_interest_score", "sum"),
                last_timestamp=("last_timestamp", "max"),
                yes_rate=("yes_rate", "mean"),
            )
            .reset_index()
        )
        category_rows["category_ml"] = predicted_category

    category_rows = category_rows.rename(
        columns={
            "yes_rate": "category_yes_rate",
            "total_interest_score": "recent_interest_score",
            "interactions": "category_interactions",
        }
    )

    merged = category_rows.merge(user_profiles, on="user_id", how="left")
    merged["positive_swipe_count"] = merged["positive_swipe_count"].fillna(0).astype(int)

    item_vector = normalize(item_embedding.reshape(1, -1))[0]
    merged["similarity_raw"] = merged["profile_embedding"].apply(
        lambda profile: float(np.dot(profile, item_vector)) if isinstance(profile, np.ndarray) else 0.0
    )

    merged["similarity_score"] = scale_zero_one(merged["similarity_raw"])
    merged["category_yes_rate"] = merged["category_yes_rate"].fillna(0.0)
    merged["recent_interest_score"] = merged["recent_interest_score"].fillna(0.0)
    merged["recent_interest_score"] = scale_zero_one(merged["recent_interest_score"])
    merged["history_score"] = scale_zero_one(np.log1p(merged["positive_swipe_count"]))

    merged["match_score"] = (
        0.40 * merged["category_yes_rate"]
        + 0.35 * merged["similarity_score"]
        + 0.15 * merged["recent_interest_score"]
        + 0.10 * merged["history_score"]
    )
    merged["confidence_score"] = (100 * merged["match_score"]).round(1)
    merged["why_match"] = merged.apply(
        lambda row: summarize_why_match(row, predicted_category),
        axis=1,
    )

    return merged.sort_values(
        ["match_score", "category_yes_rate", "similarity_score"],
        ascending=False,
    ).reset_index(drop=True)


uploaded_file = st.file_uploader("Upload a fashion image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file:
    original = Image.open(uploaded_file).convert("RGB")
    views = build_inference_views(original)
    embeddings = [pil_to_embedding(view) for view in views.values()]
    scores, pred = aggregate_scores(embeddings)
    label = le.inverse_transform(pred)[0]
    predicted_embedding = normalize(np.vstack(embeddings).mean(axis=0, keepdims=True))[0]

    st.subheader("Prediction")
    st.success(f"Predicted category: {label}")

    cols = st.columns(len(views) + 1)
    cols[0].image(original, caption="uploaded_image", use_column_width=True)
    for idx, (name, image) in enumerate(views.items(), start=1):
        cols[idx].image(image, caption=name, use_column_width=True)

    st.subheader("Debug")
    st.write("Embedding shape per view:", embeddings[0].shape)
    st.write("Classes:", list(le.classes_))
    st.write("Predicted index:", int(pred[0]))
    st.write("Predicted label:", label)

    top_rows = top_class_table(scores)
    if top_rows:
        st.write("Top 3 class scores:")
        st.table(
            {
                "label": [row[0] for row in top_rows],
                "score": [round(row[1], 4) for row in top_rows],
            }
        )

    matched_users = match_users_to_item(predicted_embedding, label)
    st.subheader("Users likely interested")
    st.caption("Confidence is a heuristic score based on category preference, recency-weighted behavior, and embedding similarity.")
    st.dataframe(
        matched_users[
            [
                "user_id",
                "confidence_score",
                "category_yes_rate",
                "similarity_raw",
                "positive_swipe_count",
                "why_match",
            ]
        ].head(TOP_USER_COUNT),
        use_container_width=True,
    )

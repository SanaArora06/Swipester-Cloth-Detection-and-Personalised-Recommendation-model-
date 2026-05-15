This project addresses two key challenges in thrifting, which are accurate personalisation from noisy real-world images and lack of trust in thrift marketplaces due to uncertain garment quality. These problems impact user engagement, reduce returns, and undermine Swipster's platform credibility. The solution has applications in e-commerce platforms, resale marketplaces, and automated moderation systems, with the potential to improve recommendation quality, enhance trust, and streamline quality control.

For personalisation, we developed a multimodal machine learning pipeline using EfficientNet image embeddings combined with captions, metadata, and attribute labels. Models including logistic regression, LinearSVC, boosting, and soft-voting ensembles were used, as they handle high-dimensional features and improve robustness. CNN fine-tuning was also explored for feature refinement. For quality classification (SwipesterQC-v2), we used engineered image features and an ensemble model with probability calibration and edge-tear detection.


Key challenges included class imbalance, noisy backgrounds, incomplete annotations, and limited defect datasets. These were addressed using preprocessing, segmentation-aware inputs, dataset merging, and feature fusion.

The classifier achieved ~86% accuracy, while the recommendation system reached a recall@3 of 94.29%. The quality model achieved 91.03% accuracy with a macro-F1 of 0.909, indicating strong real-world performance.

The system is deployable via a Streamlit interface with backend APIs and can scale with improved data pipelines, though challenges such as latency, dataset drift, and real-time updates remain.

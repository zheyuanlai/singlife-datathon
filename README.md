# Agent-Client Matching Model

## Overview
This project is developed as part of the NUS Datathon 2025, in collaboration with Singlife, to recommend financial advisors (agents) to customers using a data-driven approach. The model employs a multi-stage pipeline involving clustering, classification, and ranking techniques to optimize agent-client matching, aiming to enhance policy adoption and business efficiency.

## Installation
To set up the environment, install the required dependencies using:

```sh
pip install -r requirements.txt
```

Ensure you have Python 3.8+ installed.

## Dataset Overview
The project utilizes a structured dataset integrating multiple sources:
- **Client Information**: Demographic, economic, and behavioral data.
- **Policy Information**: Details about insurance policies, including premium values and policy status.
- **Agent Information**: Financial advisors' experience, expertise, and historical success rates.
- **Merged Data**: A consolidated dataset combining all relevant features for model training.

### Key Data Challenges
- **High Dimensionality**: Over 60 features require dimensionality reduction and feature selection.
- **Imbalanced Agent-to-Customer Ratio**: The dataset contains significantly more customers than agents.
- **Absence of Negative Examples**: All recorded policy purchases are successful, requiring alternative ways to define unsuccessful matches.
- **Lack of Clear Match Quality Metric**: Match effectiveness is not explicitly provided and must be inferred.
- **Fairness Considerations**: Ensuring unbiased agent recommendations across demographics.

**Note:** The Singlife dataset used in this project is proprietary and not open-source.

## Methodology
The project explores two primary approaches for agent-client matching:

### **1. Clustering-Classification-Ranking Pipeline**
1. **Preprocessing & Feature Engineering**
   - Missing values handled, categorical variables encoded, and date fields processed.
   - Dimensionality reduced using Multiple Correspondence Analysis (MCA).
2. **Agent Clustering (GMM)**
   - Agents are grouped using a Gaussian Mixture Model to capture latent similarities.
3. **Client Classification (Random Forest)**
   - Clients are classified into agent clusters using a Random Forest classifier.
4. **Agent Ranking (XGBoost Ranker)**
   - Within each cluster, agents are ranked based on historical success metrics and policy conversion rates.

### **2. Representation Learning-Based Matching** (Future Extension)
1. **Transformer-Based Embeddings**
   - Agent and customer features are converted into embeddings using a deep learning model.
2. **Cosine Similarity Ranking**
   - Clients are matched to agents based on similarity scores in the learned representation space.
3. **Contrastive Learning Fine-Tuning**
   - Supervised fine-tuning refines the model to better differentiate good and bad matches.

## Evaluation Metrics
### **1. Clustering Performance**
- **Silhouette Score**: Measures cohesion and separation.
- **Davies-Bouldin Index**: Assesses cluster compactness.
- **Log-Likelihood**: Evaluates model fit.

### **2. Classification Performance**
- **Accuracy, Precision, Recall, F1-score**: Standard classification metrics to assess model effectiveness.

### **3. Ranking Performance**
- **NDCG@5 (Normalized Discounted Cumulative Gain)**: Evaluates ranking quality.
- **Mean Reciprocal Rank (MRR)**: Assesses the position of the first relevant agent.

## Usage
Run the main logic using `main.ipynb`. The notebook includes:
1. Data preprocessing steps.
2. Training and evaluation of the clustering-classification-ranking pipeline.
3. Generating agent recommendations based on trained models.

## Results & Insights
- The classification model achieved **80.67% accuracy** in predicting agent-client compatibility.
- The ranking model showed **perfect NDCG@5** but low **MRR**, indicating the need for better ranking refinement.
- Fairness constraints were applied to ensure unbiased recommendations while maintaining model interpretability.
- Business implications suggest that **workload balancing among agents** can enhance efficiency.

## Future Work
- **Improve Clustering Methods**: Alternative clustering techniques such as hierarchical clustering or deep clustering.
- **Refine Ranking Model**: Incorporating additional behavioral signals to improve MRR.
- **Deploy Transformer-Based Representation Learning**: Fully implement and fine-tune deep learning-based matching.
- **Enhance Real-Time Adaptability**: Develop models that dynamically adjust recommendations based on new data.

## Contributors
This project is developed by Zheyuan Lai, Yuxin Liu, Xiyao Ma, Jingyu Shi, and Yuhan Wang as part of the NUS Datathon 2025. All contributors have contributed equally.

For any queries, reach out at [zheyuan_lai@u.nus.edu](mailto:zheyuan_lai@u.nus.edu).
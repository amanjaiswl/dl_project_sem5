# Smart Product Pricing: A Multi-Modal Machine Learning Solution

**Project:** E-Commerce Price Prediction Challenge (2025)
**Goal:** Predict optimal product prices based on textual descriptions and product images.
**Evaluation Metric:** Symmetric Mean Absolute Percentage Error (SMAPE).

---

## 1. Executive Summary

This project explores the evolution of pricing algorithms, moving from traditional statistical machine learning to state-of-the-art Multi-Modal Large Language Models (LLMs). We implemented and evaluated three distinct architectures to solve the challenge of predicting prices from unstructured data.

Our final solution, a **Multi-Modal Fusion System** utilizing fine-tuned **Qwen3 (LLM)** and **CLIP/SigLIP (Vision)** encoders with a **Mixture-of-Experts (MoE)** regression head, achieved our best performance with a SMAPE of **44.0%**, significantly outperforming the baseline.

| Approach | Architecture | Key Technologies | SMAPE Score |
| :--- | :--- | :--- | :--- |
| **1. Baseline** | Gradient Boosting | LightGBM, TF-IDF, SVD | **50.7%** |
| **2. Intermediate** | Hybrid Neural Network | DistilBERT, ResNet18, Concatenation | **48.0%** |
| **3. Final (SOTA)** | Foundation Model Fusion | Qwen3-QLoRA, CLIP, SigLIP, MoE | **44.0%** |

---

## 2. Approach 1: The Machine Learning Baseline (LightGBM)

### Overview
Our initial approach focused on robust **Feature Engineering** applied to tabular data. Instead of raw deep learning, we manually extracted insights from the text and fed them into a highly efficient Gradient Boosting Decision Tree (GBDT).

### Key Methodologies
* **Text Processing:** Implemented `enhanced_tokenizer` to standardize units (e.g., "lbs" -> "pounds") and regex extractors to capture **Item Pack Quantities (IPQ)** (e.g., "Pack of 12").
* **Dimensionality Reduction:** Used **TF-IDF** to vectorize text, followed by **Truncated SVD (50 components)** to capture semantic concepts without exploding feature space.
* **Target Transformation:** Applied `log1p(price)` to handle the heavy right-skew of price distributions.
* **Model:** **LightGBM** trained with 5-fold cross-validation.

### Analysis
While fast and interpretable, this model struggled with the "semantic gap." It could count the word "Premium," but it didn't understand that "Premium" implies a higher price.
* **Final SMAPE:** 50.7%

---

## 3. Approach 2: Hybrid Deep Learning (Custom CNN + BERT)

### Overview
To capture the semantic meaning of text and the visual signals of product quality, we moved to a **Deep Learning** approach. We designed a custom neural network that processes text and images in parallel branches before fusing them.

### Key Methodologies
* **Text Branch:** Utilized **DistilBERT** (frozen early layers) to generate 768-dimensional embeddings from the raw catalog descriptions.
* **Vision Branch:** Utilized **ResNet18** (pre-trained on ImageNet) to extract visual features from product images (resized to 224x224).
* **Feature Fusion:** Concatenated the BERT embeddings, ResNet embeddings, and manual features (from Approach 1) into a single vector.
* **Loss Function:** Switched to **Huber Loss**, which is more robust to outliers than Mean Squared Error.

### Analysis
This approach improved performance by leveraging Transfer Learning. However, the naive concatenation of features meant the model struggled to correlate specific visual cues with specific textual claims.
* **Final SMAPE:** 48.0%

---

## 4. Approach 3: State-of-the-Art Multi-Modal Fusion (Final Solution)

### Overview
Our final solution leverages **Foundation Models**. Instead of training feature extractors from scratch, we fine-tuned "Expert" models and used advanced architectural patterns to fuse their knowledge.

### Architecture
1. **Fine-Tuned LLM (Text):**
   * **Model:** **Qwen3-0.6B**.
   * **Technique:** **QLoRA** (Quantized Low-Rank Adaptation). We fine-tuned the LLM specifically on the pricing regression task, allowing it to understand e-commerce nuance better than generic BERT.
2. **Dual-Vision Encoders:**
   * **CLIP:** Captures general semantic categories.
   * **SigLIP:** Captures fine-grained visual details.
3. **Hierarchical Fusion:**
   * Instead of simple concatenation, we used **Cross-Attention** layers. This forces the Text and Image embeddings to "attend" to each other (e.g., aligning the text "12-pack" with the visual of 12 bottles) before making a prediction.
4. **Mixture of Experts (MoE):**
   * We replaced the single prediction head with **3 Expert Networks** controlled by a Gating Network.
   * This allowed different neural networks to specialize in different price ranges (Low, Mid, High).

### Strategic Optimization
* **Weighted SMAPE Loss:** We implemented a tiered loss function that penalized errors on cheap items (under $5) **10x** more than expensive items. This directly optimized the model for the SMAPE metric, which is highly sensitive to errors in low-value products.

### Analysis
This approach yielded significant gains, particularly in the difficult $0-$30 price range, proving that foundation models and specialized architectures are essential for high-precision pricing.
* **Final SMAPE:** 44.0%

---

## 5. Conclusion

The progression from LightGBM to Multi-Modal Fusion highlights the importance of **semantic understanding** in pricing. While manual features (Approach 1) provide a solid floor, the ability of LLMs (Approach 3) to reason about product descriptions and the ability of MoE architectures to handle diverse price scales provided the necessary edge to minimize error.


# Brain Tumor Classification from MRI Images

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow.svg)

</div>

## 🧠 Project Overview

This comprehensive machine learning project tackles the critical medical challenge of **brain tumor classification** from MRI images. The project implements and compares multiple state-of-the-art machine learning approaches, achieving **95% accuracy** through advanced deep learning techniques and innovative hierarchical classification methods.

### 🎯 Objective
Develop an automated system to classify brain MRI scans into four categories:
- **No Tumor** (healthy brain)
- **Glioma Tumor** (malignant, 30% of brain tumors)
- **Meningioma Tumor** (benign, slow-growing)
- **Pituitary Tumor** (affects hormone levels)

## 🌟 Key Achievements

- ✅ **95% Classification Accuracy** using CNN with data augmentation
- 🔬 **Comprehensive Comparison** of 8+ machine learning algorithms
- 🏗️ **Novel Hierarchical Classification** approach achieving 94% accuracy
- 📊 **Advanced Feature Engineering** with image preprocessing techniques
- 🎛️ **Hyperparameter Optimization** using RandomizedSearchCV
- 🔄 **Cross-Validation** analysis with StratifiedKFold

## 🚀 Technical Highlights

### Machine Learning Models Implemented
1. **Convolutional Neural Networks (CNN)**
   - Custom architecture with batch normalization
   - Early stopping and model checkpointing
   - Data augmentation (rotation, flipping, Gaussian blur)

2. **Traditional ML Algorithms**
   - Random Forest (90%+ accuracy after tuning)
   - Support Vector Machine with RBF kernel
   - K-Nearest Neighbors
   - Decision Trees
   - Logistic Regression
   - Naive Bayes

3. **Innovative Approaches**
   - **Hierarchical Classification**: Multi-stage classification system
   - **Binary Classifier Chains**: Tumor detection → Type classification
   - **Ensemble Methods**: Combining multiple model strengths

### Advanced Techniques
- **Image Preprocessing**: Gaussian filtering, normalization, resizing
- **Data Augmentation**: Vertical flip, 90° rotation, deterministic augmentation
- **Regularization**: Dropout, batch normalization, L1/L2 regularization
- **Cross-Validation**: 5-fold stratified validation for robust evaluation

## 📊 Results Summary

| Approach | Accuracy | Key Strengths |
|----------|----------|---------------|
| CNN + Data Augmentation | **95%** | Best overall performance, robust to variations |
| Hierarchical CNN | **94%** | Excellent tumor detection, interpretable pipeline |
| Random Forest (Tuned) | **90%** | Fast inference, good feature importance |
| SVM (RBF) | **87%** | Strong separation boundaries |
| Hierarchical ML | **87%** | Combines multiple algorithm strengths |

### Detailed Performance Analysis
- **Precision**: Up to 100% for specific tumor types (Pituitary)
- **Recall**: Excellent tumor detection with minimal false negatives
- **F1-Score**: Balanced performance across all classes
- **Cross-Validation**: Consistent performance across different data splits

## 🛠️ Technical Implementation

### Dataset Characteristics
- **3,264 MRI images** from Kaggle brain tumor dataset
- **JPEG format**, RGB converted to grayscale
- **128x128 pixel** resolution after preprocessing
- **Quality filtering**: Removed 69 images with poor aspect ratios

### Image Preprocessing Pipeline
```python
# Key preprocessing steps implemented:
1. Quality assessment and filtering
2. Grayscale conversion
3. Gaussian blur noise reduction
4. Pixel normalization (0-1 range)
5. Augmentation (rotation, flipping)
6. Train/test split with stratification
```

### CNN Architecture Highlights
- **Input**: 128×128×1 grayscale images
- **Layers**: 4 convolutional blocks with max pooling
- **Features**: 32→64→128→128 filters
- **Regularization**: Dropout (0.25), Batch Normalization
- **Output**: 4-class softmax classification
- **Optimization**: Adam optimizer with categorical crossentropy

### Hierarchical Classification Innovation
The project introduces a novel **3-stage hierarchical approach**:
1. **Stage 1**: Tumor vs. No Tumor detection (99%+ accuracy)
2. **Stage 2**: Pituitary vs. Other tumor types (98%+ precision)
3. **Stage 3**: Glioma vs. Meningioma classification (94% accuracy)

This approach mimics radiologist decision-making processes and provides interpretable intermediate results.

## 🔬 Research Methodology

### Experimental Design
- **Controlled comparisons** across multiple algorithms
- **Ablation studies** on preprocessing techniques
- **Hyperparameter optimization** using grid/random search
- **Statistical validation** with cross-validation
- **Performance visualization** with confusion matrices and learning curves

### Key Insights
1. **Data quality matters**: Filtering poor-quality images improved all models
2. **Augmentation is crucial**: 3x dataset expansion through rotation/flipping
3. **Architecture optimization**: Batch normalization significantly improved training stability
4. **Hierarchical advantages**: Breaking down complex problems improves interpretability

## 📁 Repository Structure

```
Brain_Tumor_Classification/
├── brain_tumor_classification.ipynb    # Main analysis notebook
├── README.md                          # This comprehensive documentation
├── Hierarchical Classification Diagram.png  # Visual workflow
└── requirements.txt                   # Dependencies (if needed)
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow>=2.0
pip install scikit-learn>=0.24
pip install opencv-python
pip install matplotlib seaborn
pip install jupyter notebook
pip install tqdm
```

### Running the Analysis
1. **Clone the repository**
2. **Download the dataset** from [Kaggle Brain Tumor Classification](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)
3. **Open the Jupyter notebook**: `brain_tumor_classification.ipynb`
4. **Run cells sequentially** to reproduce results

## 🎓 Educational Value

This project demonstrates mastery of:

### Machine Learning Concepts
- **Multi-class classification** in medical imaging
- **Model comparison** and selection criteria
- **Overfitting prevention** and regularization
- **Performance evaluation** metrics and interpretation

### Technical Skills
- **Deep learning** with TensorFlow/Keras
- **Computer vision** with OpenCV
- **Data science** workflow and best practices
- **Statistical analysis** and validation techniques

### Professional Practices
- **Comprehensive documentation** and reproducible research
- **Medical domain knowledge** integration
- **Ethical considerations** in healthcare AI
- **Performance optimization** and computational efficiency

## 🔮 Future Enhancements

### Immediate Improvements
- [ ] **Larger dataset** integration for improved generalization
- [ ] **Advanced augmentation** techniques (mixup, cutmix)
- [ ] **Transfer learning** from pre-trained medical imaging models
- [ ] **Segmentation preprocessing** to isolate tumor regions

### Advanced Research Directions
- [ ] **Explainable AI** integration (GradCAM, LIME)
- [ ] **Uncertainty quantification** for clinical confidence
- [ ] **Multi-modal fusion** (combining different MRI sequences)
- [ ] **Real-time inference** optimization for clinical deployment

## 👨‍💻 Authors

**Gilad Battat** & **Doron Gaznavi**  
*Computer Science Students, The Open University of Israel*

## 📊 Impact & Applications

### Medical Significance
- **Early detection** capabilities for improved patient outcomes
- **Standardized screening** reducing dependency on specialist availability
- **Second opinion** tool for radiologists
- **Telemedicine** applications for remote diagnosis

### Technical Contributions
- **Benchmark comparison** of ML algorithms on medical imaging
- **Hierarchical classification** methodology for multi-class problems
- **Preprocessing pipeline** optimized for MRI data
- **Open-source implementation** for reproducible research

---

<div align="center">

*This project represents cutting-edge application of machine learning to critical healthcare challenges, demonstrating both technical excellence and practical impact.*

</div>


# **MatPropNet: A Machine Learning Framework for Zeolite Property Prediction**

## **Overview**

MatPropNet is a machine learning framework designed for predicting various properties of materials, more specifically zeolites, using neural networks. Built with **TensorFlow** and **Keras**, it supports **cross-validation**, **hyperparameter tuning**, and **model evaluation**. The framework's modular structure allows easy customization and experimentation through configuration files.

The datasets used in this framework were sourced from my [Web Scraping project](https://github.com/mfaria-p/Webscrapping_zeolites.git) on zeolites conducted as part of an Introduction and Initiation Research Internship at the Laboratory of Separation and Reaction Processes - Laboratory of Catalysis and Materials (LSRE-LCM). This project created a comprehensive database based on data from the [International Zeolite Association (IZA)](https://www.iza-structure.org/databases/), capturing key properties of 251 zeolites, including unit cell shapes, framework density, ring sizes, channel dimensions, and more. Through Python techniques, I used the composite building units of the zeolites to generate the fingerprints, which were given as input for the neural network.

---

## **Key Features**

- **Cross-Validation**: Ensures robust model evaluation.
- **Hyperparameter Tuning**: Leverages **Keras Tuner** for optimized model performance.
- **Model Training and Evaluation**: Simplified scripts for training, testing, and evaluating models, with features like saving model history and generating parity plots.
- **Configuration-Driven**: Uses configuration files to specify model architecture, training parameters, and data paths for easy experimentation.

---

## **Directory Structure**

```
MatPropNet/
|├── config_files/       # Configuration files for different models and datasets
|├── data/              # Datasets used for training and testing
|├── hyperband/         # Contains the results of hyperparameter tuning
|├── main/              # Main scripts for training, testing, and evaluating models
|   |├── data.py            # Functions for loading and preprocessing data
|   |├── main_cv.py         # Script for cross-validation
|   |├── main_hyper.py      # Script for hyperparameter tuning
|   |├── model.py           # Functions for building, training, and testing models
|   |├── saving.py          # Functions for saving model history
|   └── stats.py           # Utility functions for statistical analysis
|└── models/            # Saved models and their associated metadata
```

---

## **Getting Started**

### **1. Install Dependencies**

Ensure you have **TensorFlow**, **Keras**, and other required libraries installed.

### **2. Prepare Data**

Place your datasets in the **data/** directory. The datasets provided were created from my [Zeolite Web Scraping project](https://github.com/mfaria-p/Webscrapping_zeolites.git), and are composed of **fingerprints** that were generated based on the composite building units of each zeolite, through Python techniques.

### **3. Configure Models**

Edit the configuration files in the **config_files/** directory to specify model architecture, training parameters, and data paths.

### **4. Run Scripts**

Use the scripts in the **main/** directory to train, evaluate, and tune your models.

---

## **Example Usage**

### **To Run Cross-Validation:**

```bash
python3 main/main_cv.py config_files/density_iza.cfg
```

### **To Perform Hyperparameter Tuning:**

```bash
python3 main/main_hyper.py config_files/density_iza.cfg
```




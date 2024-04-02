![](assets/CustomerInsightsCrop.jpg)
# Customer Insights
> Project Group 33: Sidhant Subramanian, Sreenidhi Reddy Bommu, Bharat Raghunathan, Ayushi Mathur, Vishnu Varma Venkata

## Midterm
### Introduction/Background

Using machine learning (ML) in marketing revolutionises personalised strategies by analysing customer traits. This project aims to harness ML to delve into these traits, enhancing targeted marketing and customer relationships. The untapped potential of ML in extracting and correlating traits from vast datasets offers promising prospects for marketing precision and efficiency.

Using machine learning (ML) in marketing has revolutionised the way businesses approach personalised strategies, fundamentally changing the landscape of customer engagement and relationship management.

By leveraging ML algorithms, marketers can now analyse complex datasets to identify intricate customer traits and preferences that are not immediately apparent., enabling a level of personalization that was previously unattainable.

This capability is particularly valuable in marketing, where understanding the nuances of customer behaviour can lead to more effective and personalised engagement strategies.

### Literature Review

Machine learning is crucial for personalized marketing, predicting customer personalities from digital footprints [1], enhancing engagement and sales with personality-tailored advertising [2], and enabling deeper personalization through advanced segmentation techniques based on personality traits [3].

### Dataset Description and Link

The Kaggle "Customer Personality Analysis" dataset is used to predict consumer behavior and improve marketing strategies, enhancing business engagement and expenditure optimization.

Dataset Link: [Customer Personality Analysis Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data)

### Problem and Motivation

#### Problem
Develop a machine learning model to classify customers based on their likelihood to purchase products from specific categories or channels.

This classification should consider customers' demographic details (age, education, marital status, income), household composition (number of children and teenagers), engagement history (website visits, recency of purchase), and purchasing behaviour (amount spent on various product categories, number of purchases through different channels, responses to marketing campaigns)


#### Motivation
The goal is to identify distinct customer segments that prefer different shopping venues (online, in-store, catalog) and product categories (wines, fruits, meats, fish, sweets, gold products), enabling personalized marketing strategies. This model will support targeted marketing initiatives by predicting the preferred shopping channels and product categories for new and existing customers, thereby increasing customer engagement and optimizing marketing spend.

### Methods

#### Preprocessing

#### Feature encoding
Transforming categorical variables into other representations using one-hot encoding or label encoding since some models require numerical input.

#### Feature scaling
Technique applied to make the input features of the dataset belong to similar ranges to ensure all features have the same effect on the model.

#### Dimensionality reduction 
Transforming the dimensionality of features from higher to lower while retaining most of the information like **PCA** or **LDA**. Higher dimensionality creates a complex model which leads to overfitting, requiring higher computation and reduced model interpretability.

#### Correlation
Dropping highly correlated features to avoid multicollinearity.

#### Supervised Learning Algorithms

#### Linear SVM
Separates the data using a single linear decision boundary. It is computationally cheaper and faster to train on.

#### Random Forest
Separates the data with a non-linear decision boundary and takes into account interactions between variables. They are known to be robust and resistant to overfitting.

#### Neural Network
Separates the data by non-linear decision boundary and handles large amounts of unstructured data as input. They model complex patterns between data and can accommodate diverse data distributions as input.

#### Unsupervised Learning Algorithms

#### K - Means
Clustering algorithm using centroids which works best for evenly sized clusters.


#### Hierarchical Clustering
Works when the data can be divided hierarchically based on merging samples successively. Doesn't require us to specify number of clusters.

#### OPTICS
Clustering algorithms based on density which allows for cluster sizes and density being arbitrary. It is robust to noise and doesn't require specifying number of clusters.

### Results and Discussion

#### Supervised Learning Metrics

1. Accuracy
2. Precision
3. Recall
4. F-1 score

#### Unsupervised Learning Metrics

1. Elbow method
2. Silhouette coefficient

**Project Goals:** To develop a classification of customers into places they are most likely to purchase from.

**Expected Results:** An accurate and robust model assisting in targeted marketing for increased sales and customer retention.

### Proposal Timeline
[Link to Gantt Chart Spreadsheet](https://gtvault-my.sharepoint.com/:x:/g/personal/braghunathan6_gatech_edu/EdOIA96B63lAuuimPIdRCZ4BP_hUtCNmEp74v8O8sn5kRA?e=7HXVFD)
![Gantt Chart](assets/GanttChart.png)

### Contribution Table

| Name    | Proposal Contribution              |
|:---------|:-----------------------------------|
| Sreenidhi Reddy Bommu | Introduction, Literature Review |
| Ayushi Mathur | Problem Definition, Motivation   |
| Sidhant Subramanian | Methods, Results and Discussion |
| Bharat Raghunathan | GitHub Repository, GitHub Pages, Gantt Chart |
| Vishnu Varma Venkata | Report, Video, Presentation  |

### References
[1]: <a>https://doi.org/10.1073/pnas.1218772110</a> - `M. Kosinski, D. Stillwell, and T. Graepel, "Private traits and attributes are predictable from digital records of human behavior," Proceedings of the National Academy of Sciences, vol. 110, no. 15, pp. 5802-5805, April 2013.`

[2]: <a>https://www.pnas.org/doi/full/10.1073/pnas.1710966114</a> - `S.C. Matz, M. Kosinski, G. Nave, and D.J. Stillwell, "Psychological targeting as an effective approach to digital mass persuasion," Proceedings of the National Academy of Sciences, vol. 114, no. 48, pp. 12714-12719, November 2017.`

[3]: <a>https://doi.org/10.48550/arXiv.2306.17170</a> - `J. Zhang, Y. Wang, and P.S. Yu, "Community structure detection in social networks with node attributes," IEEE Transactions on Knowledge and Data Engineering, vol. 31, no. 10, pp. 1984-1997, Oct. 2019.`

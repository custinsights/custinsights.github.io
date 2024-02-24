![](assets/CustomerInsightsCrop.jpg)
# Customer Insights
> Project Group 33: Sidhant Subramanian, Sreenidhi Reddy Bommu, Bharat Raghunathan, Ayushi Mathur, Vishnu Varma Venkata

## Proposal
### Introduction/Background

The integration of machine learning (ML) in digital marketing revolutionises personalised strategies by analysing customer personality traits. This project aims to harness ML to delve into these traits, enhancing targeted marketing and customer relationships. The untapped potential of ML in extracting and correlating personality traits from vast datasets offers promising prospects for marketing precision and efficiency.


### Literature Review

Machine learning's role in decoding customer personalities is pivotal for personalised marketing. Kosinski, Stillwell, and Graepel [1] showcased ML's ability to predict personal attributes from digital footprints, establishing a basis for personality-driven marketing strategies. 

Matz et al. [2] further validated the efficacy of personality-tailored advertising in boosting engagement and sales, highlighting ML's capacity to resonate marketing with individual personalities. 

Zhang, Wang, and Yu [3] demonstrated advanced segmentation techniques, emphasising ML's utility in achieving deeper marketing personalization through customer segmentation based on personality traits.


### Dataset Description and Link

Utilising the "Customer Personality Analysis" dataset from Kaggle, this project will explore demographic, purchase history, and marketing response data to unveil personality traits and predict consumer behaviour. This analysis aims to equip businesses with insights for refining marketing strategies, ultimately enhancing engagement and optimising expenditures.

Here's the link to the [Customer Personality Analysis Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data).

### Problem and Motivation

✔️ **Problem**: Classify customers based on the places they are most likely to buy products from.

✔️ **Motivation**: Customer personality analysis can help a business to modify their products and their marketing strategies according to the customers’ traits. Based on the analysis of what products are being purchased from what places, the marketing and selling strategy of a business can be devised effectively.

### Methods

#### Preprocessing

**Feature encoding:**
Transforming categorical variables into other representations using one-hot encoding, label encoding, or target encoding since a lot of models require numerical input.

**Feature scaling:** 
Technique applied to make the input features of the dataset belong to  similar ranges and to ensure all features have the same effect on the model.

**Dimensionality reduction:** 
Transforming the dimensionality of features from higher to lower while retaining most of the information like Principal component analysis (PCA) or Linear discriminant analysis(LDA). We use this technique since higher dimensionality features risks creating a complex model which leads to overfitting, they also require higher computation and reduce model interpretability.

**Correlation:**
Dropping highly correlated features to avoid multicollinearity.

#### Supervised Learning Algorithms

#### Linear SVM
This algorithm can separate the data using a single linear decision boundary. It is computationally less expensive and relatively faster to train on.
Random Forest
This algorithm can separate the data with a non-linear decision boundary and can take into account interactions between variables. They are known to be robust and resistant to overfitting.

#### Neural Network
This algorithm can separate the data by non-linear decision boundary and can handle large amounts of unstructured data as input. They can model complex patterns between data and can accommodate diverse data distributions as input.

#### Unsupervised Learning Algorithms

#### K - Means
A clustering algorithm based on Centroids which works best for clusters of even size. 


#### Hierarchical Clustering
Works when the data can be divided hierarchically based on merging samples successively. Does not require us to specify the number of clusters

#### OPTICS
A clustering algorithm based on density which allows for cluster sizes and density to be arbitrary. It is robust to noise and does not require a number of clusters to be specified.

### Results and Discussion

#### Supervised Learning Metrics

1. Accuracy
2. Precision
3. Recall
4. F-1 score

#### Unsupervised Learning Metrics

1. Elbow method
2. Silhouette coefficient 
3. Calinski-Harabasz Index

**Project Goals:** To develop a classification  of customers into places they are most likely to purchase from.

**Expected Results:** An accurate and robust model to assist in targeted marketing for increased sales and customer retention.

### Proposal Timeline

### Contribution Table
| Name    | Proposal Contribution              |
|---------|-----------------------------------|
| Sreenidhi Reddy Bommu | Introduction, Literature Review |
| Ayushi Mathur | Problem Definition, Motivation   |
| Sidhant Subramanian | Methods, Results and Discussion |
| Bharat Raghunathan | GitHub Repository, Gantt Chart |
| Vishnu Varma Venkata | Report, Video, Presentation  |

### References
[1]: https://doi.org/10.1073/pnas.1218772110 - `M. Kosinski, D. Stillwell, and T. Graepel, "Private traits and attributes are predictable from digital records of human behavior," Proceedings of the National Academy of Sciences, vol. 110, no. 15, pp. 5802-5805, April 2013.`

[2]: https://www.pnas.org/doi/full/10.1073/pnas.1710966114 - `S.C. Matz, M. Kosinski, G. Nave, and D.J. Stillwell, "Psychological targeting as an effective approach to digital mass persuasion," Proceedings of the National Academy of Sciences, vol. 114, no. 48, pp. 12714-12719, November 2017.`

[3]: https://doi.org/10.48550/arXiv.2306.17170 - `J. Zhang, Y. Wang, and P.S. Yu, "Community structure detection in social networks with node attributes," IEEE Transactions on Knowledge and Data Engineering, vol. 31, no. 10, pp. 1984-1997, Oct. 2019.`

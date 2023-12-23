# Research Findings

## Research Question
Starting inspiration behind this Natural Language Processing project involved attempting to develop a tool that would be useful for chefs who like to cook but have a hard time selecting recipes as well as a technique to reduce food waste. To aid in the process, our investigation began with conducting an exploratory data analysis to predict ratings and categories of recipes because this information is useful for recipe generation analysis when such variables are not available. 

## EDA
Two datasets were used throughout the following analysis. One was a hugging face dataset called Recipe_NLG which is a cooking recipe dataset for semi-structured text generation. This dataset consisted of 2,231,142 rows. The second is an All Recipes API scrape from Kaggle. This 1,798 row dataset contains categorization and nutritional facts relating to All Recipe data. 

A combination approach was utilized because of the difference in variables between the two datasets. Recipe_NLG contained useful variables such as identified named entities of ingredients, the directions for the recipe, and the link. Key variables from the Kaggle dataset included category, rating, review count, cooking times, and nutritional information. 
We did not want to choose between these variables, but rather keep all variables for analysis. Thus a comprehensive dataset approach was built from the idea to create a consolidated dataset that encompasses a wide range of variables compared to that on one dataset alone. After cleaning and merging the data, the resulting dataset is 800 rows. 

This slimming process involved filtering the hugging face dataset to only those from All Recipes based on finding "all recipes" in the link column. After this, recipes were aligned through matching titles. We now have an all-inclusive dataset for analysis. Data consistency had to be considered to ensure uniformity in the format and structure of the merged data to facilitate easy analysis and processing.  

After this, feature engineering occurred. This process tokenizes the title and removes stop words and punctuation. Next sentiment analysis is performed on the filtered title. Bigrams for both titles and directions occur. The number of ingredients and the number of words in the directions is also discovered. The parts of speech distribution are another variable, as well as an approximation for the number of steps through the count of verbs. 

Investigating the directions through a word cloud revealed interesting insights. Words such as minute, stir, place, and cook were commonly occurring. These words are not anything out of the ordinary in the context of cooking, but the information they provide is helpful for future analysis. The word cloud results serve as a logical explanation and inspiration for the bigram_directions creation. When finding commonly occurring word phrases such as "# minutes" or "Degrees Fahrenheit" should be considered one combined token rather than separate words because of their dominant co-occurrence. In doing so, other information that has more novel insights can be discovered. 

Visualizing the top 10 bigrams of the direction column revealed that "preheat oven" and "preheated oven" were the top two. The tokens of directions go through a lemmatization process, but this insight reveals that the core component of the word preserves the word tense. Thus, they are two separate bigrams. Other top bigrams include "medium heat", "salt pepper", "reduce heat", "bake preheated", "set aside", etc. The prevalent frequency of such bigrams reveals typical language in cooking. These bigrams show a pattern of verbs followed by nouns such as "preheat oven." In general, directions reveal a lot of actions and descriptions, so these commonly occurring pairings make sense.

After exploring the categorical variables of the dataset, it is interesting to note that there is a wide distribution of state representation. The plot shows gradual decay meaning that while the states are not equally represented, there is a diminishing frequency or occurrence as one moves from one state to another. Pennsylvania, Louisiana, Washington, and Minnesota had the top representation. This insight reveals culinary diversity, as states with higher representation could indicate a richer culinary tradition or popularity in recipe contributions from those regions. This could offer insights into the diverse gastronomic culture across different states. It may allude to cultural influence as well. Variations in representation might reflect cultural influences or historical backgrounds of the regions. It could highlight the prevalence and influence of different cultural groups within each state.

Category distribution revealed a problem as there were 45 categories. Some categories had overlaps such as Everyday Cooking, Cuisine, and Main Dishes can all be grouped. Having this many categories would raise problems for classification. Therefore, categories are simplified down later into overarching categories plus a spot for others. 

Distributions of numerical categories revealed that ratings were concentrated around 4.6 with a normal distribution. Most other numerical variables such as calorie count, subjectivity, polarity, number of ingredients, number of words in the directions, and number of steps were also normally distributed. The remainder of the variables were right-skewed. 

## Recipe Matching
Recipe matching based on a list of "must-use" ingredients and additional ingredients on hand involves finding recipes that utilize the specified "must-use" ingredients while accommodating the available additional ingredients. This process aims to suggest recipes that incorporate the specified ingredients, thereby reducing food waste and utilizing existing kitchen supplies.

For instance, if a user provides a list of "must-use" ingredients such as chicken, broccoli, and rice, along with additional ingredients like bell peppers and onions, the goal is to suggest recipes that involve chicken, broccoli, and rice while also allowing for the inclusion of bell peppers and onions if possible.

The matching process involves employing various techniques such as keyword matching.

Ingredient Matching: Using algorithms to match provided ingredients with those in a recipe dataset or database. Compare ingredient lists, finding recipes that include the "must-use" ingredients.

Recipe Adaptation: Recommending recipes that not only contain the essential ingredients but also accommodate the available additional ingredients. Algorithms might rank or filter recipes based on their ability to incorporate both the required and additional ingredients.

Flexible Ingredient Substitution: Recommending recipes that allow for flexible ingredient substitutions. For instance, suggesting recipes where bell peppers can be used instead of a different variety of pepper, or where onions can replace shallots if needed.

If the matching algorithm successfully finds recipes that encompass the "must-use" ingredients along with accommodating the available additional ingredients, it serves as a valuable tool in reducing food waste, providing cooking options based on available supplies, and inspiring users to create meals without needing to purchase additional items.

The performance of such a matching system could be evaluated based on its ability to suggest recipes that utilize the specified ingredients while considering the additional ones, providing users with relevant and feasible cooking options. While achieving a perfect match might be challenging due to the diverse nature of recipes and ingredients, a system that can suggest recipes accommodating most of the listed ingredients can greatly assist users in meal planning and cooking with what they have on hand.

The recipe matching process demonstrates its effectiveness when it seamlessly aligns the user's specified "must-use" ingredients with available additional ones, successfully generating a curated list of diverse and appealing recipes. When this matching process works well, it adeptly suggests a range of recipes that not only incorporate the essential ingredients but also accommodate the supplementary ones, offering users versatile cooking options. It excels in its ability to consider various ingredient combinations and recipe variations, providing users with practical and feasible meal ideas.

Future iterations of the matching process would improve functionality so that it seamlessly integrates user preferences, cooking styles, and ingredient availability, resulting in a tailored selection of recipes that inspire culinary creativity while minimizing ingredient wastage. Such a successful matching process empowers users to explore, experiment, and craft delicious meals based on their available kitchen inventory, ultimately enhancing their cooking experiences.

## Supervised Learning
### Recipe Category Classification
Recipe Category Classification is a machine learning application that involves the categorization or labeling of recipes into specific culinary categories or types. This task utilizes multiple supervised learning techniques, a subset of machine learning methods that learn patterns from labeled data to make predictions or classifications on new, unseen data. The goal is to accurately predict categories for datasets that do not have this feature predetermined. 

In this context, supervised learning algorithms are trained on a dataset of recipes where each recipe is associated with a predefined category or type (e.g., Main Dishes, Vegetarian Dishes, Side Dishes, Baked Goods and Bread, Desserts and Sweets, Breakfast Dishes, Other). These algorithms learn the relationship between various features of the recipes (such as ingredients, cooking methods, preparation steps, etc.) and their corresponding categories.

Various supervised learning techniques like Decision Trees, Random Forests, Support Vector Machines (SVM), Naive Bayes, Logistic Regression, or Neural Networks can be employed for this classification task. These algorithms learn from the labeled dataset, identifying patterns and correlations between recipe features and their assigned categories to create a model.

The process involves the following key steps:

Data Preparation: Preprocessing the recipe dataset, which might involve cleaning the data, extracting relevant features, and encoding categorical variables.

Training: Using a portion of the labeled dataset to train the machine learning models. The algorithms learn to predict the recipe category based on the provided features.

Evaluation: Assessing the trained models' performance using evaluation metrics like accuracy, precision, recall, or F1-score on a separate portion of the dataset (validation or test set) not used during training.

Prediction: Applying the trained model to new, unseen recipes to predict their categories based on their features.

The aim of Recipe Category Classification is to create a reliable and accurate model that can automatically assign appropriate categories to recipes, aiding in organizing and indexing culinary information. Such models can be employed in recipe recommendation systems, content organization on cooking websites, or for enhancing user experiences by suggesting recipes based on specific categories or preferences.

The performance of this classification was best with Logistic Regression. It had an accuracy of 50.625%. While this accuracy might not seem exceptionally high, it signifies that the model performed better than random chance with a 1/7 chance of being correct.

In the context of recipe category classification, where the categories could include various culinary types such as main dishes, sides, etc., having an accuracy of over 50% demonstrates that the model has learned certain patterns or associations within the recipe features that enable it to make better-than-random predictions.

While the accuracy might not be extremely high, a model surpassing random chance can be immensely valuable in practical applications. It can assist in automatically categorizing recipes, aiding in content organization on cooking platforms, improving search functionality, and providing users with more relevant and categorized recipe recommendations. Additionally, it provides a foundation for further model refinement and enhancement through feature engineering, hyperparameter tuning, or employing more complex algorithms to potentially improve accuracy and generalization. Furthermore, an accuracy of 50.6% suggests that the model can save time and effort for users who would otherwise manually categorize recipes. It serves as a starting point and demonstrates the feasibility of using machine learning techniques to automate recipe categorization, even though there's room for improvement. As such, this model, despite not achieving a very high accuracy, still proves to be a beneficial and practical tool in organizing and sorting recipes based on their categories.
### Recipe Rating Prediction
In the context of Recipe Rating Prediction, a variety of machine learning models were applied and evaluated to predict recipe ratings. The models used for prediction encompassed several techniques, each with its unique characteristics:

Linear Regression: A straightforward regression model that establishes a linear relationship between features and the target variable (recipe ratings in this case).

Ridge Regression: A regularization technique that mitigates overfitting in linear regression by adding a penalty term for large coefficients, thereby reducing model complexity.

Lasso Regression: Another regularization technique similar to Ridge Regression, but with the ability to shrink some coefficients to zero, essentially performing feature selection.

Elastic Net: Combines aspects of both Ridge and Lasso Regression, utilizing both L1 (Lasso) and L2 (Ridge) penalties, aiming to strike a balance between feature selection and coefficient shrinkage.

Decision Tree: A non-linear model that creates a tree-like structure by making decisions based on features, breaking down data into smaller subsets to make predictions.

Random Forest: An ensemble learning method that constructs multiple decision trees and aggregates their predictions, often known for its robustness and performance in various tasks.

Gradient Boosting: Builds multiple weak models (typically decision trees) sequentially, where each model corrects the errors of its predecessor, boosting overall predictive power.

Support Vector Machines (SVM): A supervised learning algorithm that constructs hyperplanes to separate data points into classes, also applicable for regression tasks.

K-Nearest Neighbors (KNN): A non-parametric algorithm that predicts values based on the average of the "k" closest training examples in the feature space.

Among these models, the Random Forest model emerged as the best-performing model for Recipe Rating Prediction, achieving a relatively low Mean Absolute Error (MAE) of 0.09. The low MAE suggests that the predicted ratings from the Random Forest model were close to the actual ratings given by users, demonstrating its effectiveness in accurately estimating recipe ratings.

While other models such as Gradient Boosting, Support Vector Machines, and Linear Regression might have shown promising performances as well, the Random Forest model's superior performance, with its ensemble learning approach and robustness to overfitting, makes it a preferred choice for accurately predicting recipe ratings in this scenario.

In the context of predicting recipe ratings, the task has encountered challenges due to the ratings clustering predominantly around a high value, approximately averaging at 4.6. This clustering suggests a lack of significant variability in ratings across the dataset, making it challenging for predictive models to discern subtle differences or patterns that could influence ratings. Moreover, during supervised learning attempts using various predictors such as the number of steps, encoded category, ingredient count, direction length (in words), sentiment analysis metrics (subjectivity and polarity), nutritional information, and cooking time, it was observed that none of these predictors exhibited a strong correlation with the recipe ratings. The absence of a strong correlation indicates that these factors individually might not strongly influence or explain the variation in recipe ratings, making the task of predicting ratings underperform as these predictors do not distinctly contribute to determining the perceived quality or likability of recipes.
## Unsupervised Learning
Unsupervised learning techniques like TF-IDF (Term Frequency-Inverse Document Frequency) vector spaces and BERT (Bidirectional Encoder Representations from Transformers) play crucial roles in analyzing recipe datasets. TF-IDF vectorization transforms recipes into numerical representations, capturing the importance of terms (ingredients, cooking methods) within the dataset. By employing cosine similarity or distance metrics, TF-IDF enables comparisons between recipes, identifying similarities based on ingredient compositions or textual descriptions. On the other hand, BERT, known for its contextualized word embeddings, provides nuanced and high-dimensional representations of words, sentences, or documents. In recipe analysis, BERT's contextual understanding allows for comprehensive comprehension of recipe texts, discerning complex cooking techniques and relationships between ingredients. Both TF-IDF and BERT techniques contribute to tasks such as recipe clustering, similarity identification, ingredient substitution recommendations, and even recipe generation. These unsupervised learning methods offer powerful tools for extracting meaningful insights, understanding semantic relationships, and enhancing the exploration and understanding of culinary datasets, ultimately aiding in various aspects of recipe dataset analysis.

The clustering of recipes based on titles, directions, and ingredients shows one distinct cluster when plotted in two dimensions. The other categories are mixed together, so using BERT techniques was used to see if recipe segmentation could be any more distinct. The utilization of BERT aims to uncover deeper structures or relationships within the textual data, potentially identifying latent factors that contribute to recipe similarities or distinctions. By employing BERT embeddings, the goal is to achieve a more nuanced and refined segmentation of recipes that might not be apparent from traditional two-dimensional plots, thereby enhancing the understanding of recipe similarities and enabling a more sophisticated recipe categorization or clustering process.

# Future Iterations
Future iterations of this recipe analysis project encompass a broad spectrum of enhancements aimed at refining segmentation, improving insights, and enhancing user experiences. One avenue involves the fine-tuning of advanced language models like BERT, tailoring them specifically for culinary tasks by training on expansive recipe datasets. Additionally, integrating image analysis using computer vision techniques to interpret recipe images alongside textual data could offer comprehensive insights. Engaging users through feedback mechanisms to personalize segmentation models and implementing advanced recommendation systems based on segmented clusters are pivotal steps. The project's evolution also considers contextual understanding by analyzing user-generated content and reviews. Interactive visualization tools could empower users to navigate and comprehend recipe clusters, fostering culinary exploration. Moreover, collaborative filtering techniques and community integration, enabling user sharing, rating, and commenting on recipes, would enrich the segmentation process based on collective user behaviors. Lastly, scalability and real-time updates remain crucial, ensuring adaptability to evolving culinary trends and preferences while accommodating an expanding dataset. Integrating these future iterations aims to create a sophisticated, personalized, and engaging platform for recipe analysis, recommendation, and culinary discovery.
# NLP_Recipe_Guide

# Mary Deignan and Stephanie Schefer

## Project Overview
- To accomplish for NLP Final Project    
    - **Recipe Segmentation:** 
    Recipe segmentation is a technique employed in culinary data analysis to categorize or group similar recipes together based on their inherent characteristics or ingredient compositions. This method utilizes cosine similarity scores, a mathematical measure, to assess the resemblance or proximity between different recipes within a dataset.

    - **Machine Learning:**
        - Recipe Category Classification

        Recipe Category Classification is a machine learning application that involves the categorization or labeling of recipes into specific culinary categories or types. This task utilizes multiple supervised learning techniques, a subset of machine learning methods that learn patterns from labeled data to make predictions or classifications on new, unseen data.

        In this context, supervised learning algorithms are trained on a dataset of recipes where each recipe is associated with a predefined category or type (e.g., appetizers, desserts, soups, etc.). These algorithms learn the relationship between various features of the recipes (such as ingredients, cooking methods, preparation steps, etc.) and their corresponding categories.

        Various supervised learning techniques like Decision Trees, Random Forests, Support Vector Machines (SVM), Naive Bayes, Logistic Regression, or Neural Networks can be employed for this classification task. These algorithms learn from the labeled dataset, identifying patterns and correlations between recipe features and their assigned categories to create a model.

        - Recipe Review Prediction
        
        Recipe Review Prediction is a machine learning application that involves forecasting or estimating the potential rating a recipe might receive based on various attributes. This task falls under supervised learning techniques, which utilize labeled data to train models for making predictions on unseen data.

        In this context, supervised learning algorithms are trained on a dataset of reviews where each review is associated solely with a specific rating (e.g., star ratings). These algorithms learn patterns and associations between different aspects of the reviews (such as language, textual cues, etc.) and their corresponding ratings.

        Various supervised learning techniques can be applied for this prediction task, including Regression Models like linear regression or polynomial regression, which predict numerical ratings. Classification Models such as decision trees, random forests, support vector machines (SVM), or neural networks can categorize reviews into different rating ranges.

        The procedure involves data preprocessing, model selection, training the chosen model on the labeled review dataset, validating the model's performance, and using the trained model to predict ratings for new, unseen reviews. This approach helps estimate potential ratings for recipes based on review characteristics, aiding in identifying recipes likely to receive higher or lower ratings among users.


        - Recipe Search Based on Ingredients/Recipe Recomendation System

        Develop an NLP-based recommendation system that suggests recipes to users based on the ingredients on hand that maximize review score and minimize cooking time.

        This feature allows users to input specific ingredients they have on hand, generating a list of recipes that include those ingredients. By leveraging this tool, users can make the most of the items in their pantry or fridge without having to shop for additional ingredients.

        - Finding Similar Recipes
        
        Using cosine similarity, this functionality identifies recipes that share similar characteristics, such as ingredients, flavors, or cooking methods. It enables users to explore variations or alternatives to their preferred recipes, broadening their culinary options and suggesting comparable dishes based on specific preferences.

    - **Unsupervised Learning**
        - Utilizing TF-IDF (Term Frequency-Inverse Document Frequency) alongside BERT (Bidirectional Encoder Representations from Transformers) for visualizing recipes offers a comprehensive perspective on recipe data. Although both methods provide valuable insights, TF-IDF often stands out as the superior choice for recipe visualization. TF-IDF's strength lies in its ability to weigh the significance of terms within recipes by considering their frequency across a broader dataset. This allows for the identification of key ingredients and crucial culinary elements that define a recipe's essence. While BERT excels in understanding complex linguistic nuances and contextual relationships between words, its strength might not be optimally utilized for recipe visualization purposes, as recipe data primarily focuses on specific terms and ingredients rather than nuanced language structures. Therefore, leveraging TF-IDF for recipe visualization proves more effective in representing and understanding recipes based on ingredients, facilitating the exploration of culinary similarities and differences in a more straightforward and focused manner.


## File Overview
1. **EDA.ipynb**
2. **Machine_Learning.ipynb**
3. **Unsupervised_Learning.ipynb**
4. **RecipeSegmentation.ipynb**
5. **Recipe_Generation.ipynb**

## Future Iterations/Research Ideas
#### We are researching ingredient interactions and sustainable practices relating to food diets because it can be challenging and time-consuming to research new recipes in order to enhance the recipe ideation process while reducing ingredient wastage and mirroring market preference trends.

#### Future Topics to Investigation For Further Research
- consumer preferences
    - Understanding consumer tastes, dietary preferences, cultural influences, and regional variations can help tailor recipes that resonate with the target audience.
- ingredient interactions
    -Studying how different ingredients interact chemically and functionally can lead to better formulations, improving taste, texture, and overall quality of dishes.
- food science and technology
    -Exploring advancements in food technology and techniques can aid in developing innovative methods for ingredient utilization, preservation, and enhancement of nutritional value.
- sustainability practices
    - Investigating sustainable sourcing methods, reducing food waste, and utilizing environmentally friendly ingredients align with contemporary consumer values and global sustainability goals.
- market trends and analysis
    - Researching market trends, emerging ingredients, and culinary innovations can provide insights into evolving consumer demands and competitive landscape, aiding in creating market-relevant recipes.
- culinary techniques and cooking methods
    - Studying different cooking methods and culinary techniques can optimize ingredient use and enhance flavor profiles in recipes.
- packaging and storage solutions
    - Exploring efficient packaging and storage methods can help prolong ingredient shelf life, reducing wastage and ensuring freshness.

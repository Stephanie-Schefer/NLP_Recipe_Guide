# Research Findings

## Research Question
Starting inspiration behind this Natural Language Processing project involved attempting to develop a tool that would be useful for chefs who like to cook but have a hard time selecting recipes as well as a technique to reduce food waste. To aid in the process, our investigation began with conducting an exploratory data analysis to predict ratings and categories of recipes because this information is useful for recipe generation analysis when such variables are not available. 

## EDA
Two datasets were used throughout the following analysis. One was a hugging face dataset called Recipe_NLG which is a cooking recipe dataset for semi-structured text generation. This dataset consisted of 2,231,142 rows. The second is an All Recipes API scrape from Kaggle. This 1,798 row dataset contains categorization and nutritional facts relating to All Recipe data. 

A combination approach was utilized because of the difference in variables between the two datasets. Recipe_NLG contained useful variables such as identified named entities of ingredients, the directions for the recipe, and the link. Key variables from the Kaggle dataset included category, rating, review count, cooking times, and nutritional information. 
We did not want to choose between these variables, but rather keep all variables for analysis. Thus a comprehensive dataset approach was built from the idea to create a consolidated dataset that encompasses a wide range of variables compared to that on one dataset alone. After cleaning and merging the data, the resulting dataset is 800 rows. 

This slimming process involved filtering the hugging face dataset to only those from All Recipes based on finding "all recipes" in the link column. After this, recipes were aligned through matching titles. We now have an all-inclusive dataset for analysis. Data consistency had to be considered to ensure uniformity in the format and structure of the merged data to facilitate easy analysis and processing.  

After this, feature engineering occurred. This process tokenizes the title and removes stop words and punctuation. Next sentiment analysis is performed on the filtered title. Bigrams for both titles and directions occur. The number of ingredients and the number of words in the directions is also discovered. The parts of speech distribution is another variable, as well as an approximation for the number of steps through the count of verbs. 

Investigating the directions through a word cloud revealed interesting insights. Words such as minute, stir, place, and cook were commonly occurring. These words are not anything out of the ordinary in the context of cooking, but the information it provides is helpful for future analysis. The word cloud results serve as logical explaination and inspiration for the bigram_directions creation. When finding commonly occuring word phrases such as "# minutes" or "Degrees Farenheit" should be considered one combined token rather than separate words because of thier dominant co-occurance. In doing so, other information that has more novel insights can be discovered. 

Visualizing the top 10 bigrams of the direction column revealed that "preheat oven" and "preheated oven" were the top two. The tokens of directions go through a lemmatization process, but this insight reveals that the core component of the word preserves the word tense. Thus, they are two separate bigrams. Other top bigrams include "medium heat", "salt pepper", "reduce heat", "bake preheated", "set aside", etc. The prevalent frequency of such bigrams reveals typical language in cooking. These bigrams show a pattern of verb followed by nouns such as "preheat oven." In general, directions reveal a lot of actions and descriptions, so these commonly occurring parings make sense.

After exploring the categorical variables of the dataset, it is interesting to note that there is a wide distribution of state representation. The plot shows gradual decay meaning that while the states are not equally represented, there is a diminishing frequency or occurrence as one moves from one state to another. Pennsylvania, Louisianna, Washington, and Minnesota had the top representation. This insight reveals culinary diversity, as states with higher representation could indicate a richer culinary tradition or popularity in recipe contributions from those regions. This could offer insights into the diverse gastronomic culture across different states. It may allude to cultural influence as well. Variations in representation might reflect cultural influences or historical backgrounds of the regions. It could highlight the prevalence and influence of different cultural groups within each state.

Category distribution revealed a problem as there were 45 categories. Some categories had overlaps such as Everyday Cooking, Cuisine, and Main Dishes can all be grouped together. Having this many categories would raise problems for classification. Therefore, categories are simplified down later into overarching categories plus a spot for others. 

Distributions of numerical categories revealed that ratings were concentrated around 4.6 with a normal distribution. Most other numerical variables such as calorie count, subjectivity, polarity, number of ingredients, number of words in the directions, and number of steps were also normally distributed. The remainder of the variables were right skewed. 

## Recipe Matching

## Supervised Learning

## Unsupervised Learning

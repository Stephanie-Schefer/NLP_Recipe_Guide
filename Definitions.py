import pandas as pd
import zipfile
import os
import random
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
#from transformers import BertTokenizer, BertModel

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

import re

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk import word_tokenize, pos_tag
from nltk.util import ngrams
from nltk.corpus import stopwords
import string
from collections import Counter
from textblob import TextBlob


# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

""" 
save_file(df, name)

Input:
df: DataFrame to be saved.
name: Name of the file to be saved.

Functionality:

Retrieves the current directory and creates a folder named "Data" within it if it doesn't exist.
Specifies a file name and concatenates it with the directory to create a file path.
Attempts to save the DataFrame df to a CSV file at the specified file path.
Prints a success message if the file is saved, otherwise prints an error message.
"""


def save_file(df, name):
    

    current_dir = os.getcwd()
    # if not exists make the directory
    save_folder = current_dir+'\Data'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # specify file name
    file_name = name
    
    # concatenate directory and file
    file_path = os.path.join(save_folder,file_name)
    try: 
        df.to_csv(rf"{file_path}", index = False)
        print(f"File '{file_name}' saved to '{save_folder}'")
    except Exception as e:
        print(f"Error saving file:{e}")

        
        
def analyze_recipe(title, ingredients, directions):
    # making stop words
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    # initialize WordNetLemmatizer()
    lemmatizer = WordNetLemmatizer()
    
    ######### TITLE #################
    # tokenize title into words
    words_in_title = word_tokenize(title)
    
    # remove stop words
    filtered_words_title = [word for word in words_in_title if word.lower() not in stop_words]
    
    # number of non-stop-words in title
    num_words_title = len(filtered_words_title)
    
    # making a string of the filtered title
    filtered_words_title_str = ' '.join(filtered_words_title)
    
    # perform sentiment analysis on the filtered title    
    blob = TextBlob(filtered_words_title_str)
    sentiment = blob.sentiment
    
    polarity = sentiment.polarity
    subjectivity = sentiment.polarity  
    
    # Create bigrams from the words in title
    bigrams_title = list(ngrams(filtered_words_title,2))
        
    ######### INGREDIENTS ###############
    # Number of ingredients
    num_ingredients = len(ingredients)
        
    ######## DIRECTIONS #################
    # merge directions into a single string
    merged_directions = ' '.join(directions)
    
    # tokenize merged directions into words
    words_in_directions = word_tokenize(merged_directions)    
    
    # remove stop words, punctuation, and lower case words
    filtered_words_directions = [
        word.lower() for word in words_in_directions 
        if word.lower() not in stop_words and word not in punctuation and not word.lower().startswith("'")
    ]
 
    # leematize words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words_directions]

    # normalize degrees F/C and # minutes
    normalized_directions = []
    for word in lemmatized_words:
        if re.match(r'^\d+$', word):  # If the word is a number
            next_word_index = lemmatized_words.index(word) + 1
            if next_word_index < len(lemmatized_words):
                next_word = lemmatized_words[next_word_index]
                if next_word.lower() in ['degrees', 'degree']:  # Temperature representation
                    normalized_directions.append(word + next_word.lower())
                    lemmatized_words[next_word_index] = ''  # Set next word as empty string to avoid duplication
                elif next_word.lower() == 'minutes':  # Minutes representation
                    normalized_directions.append(word + 'minutes')
                    lemmatized_words[next_word_index] = ''  # Set next word as empty string to avoid duplication
        elif word.lower() == 'degree':  # Handle singular 'degree' separately
            next_word_index = lemmatized_words.index(word) + 1
            if next_word_index < len(lemmatized_words):
                next_word = lemmatized_words[next_word_index]
                if next_word.lower() in ['fahrenheit', 'celsius','c','f']:
                    normalized_directions.append('degree' + next_word.lower())
                    lemmatized_words[next_word_index] = ''  # Set next word as empty string to avoid duplication
        elif word.lower() not in ['fahrenheit', 'celsius','c','f']:
            normalized_directions.append(word)

    # remove empty strings from the normalized directions
    normalized_directions = list(filter(None, normalized_directions))

    # Reconstruct the normalized directions into a single string
    normalized_directions_string = ' '.join(normalized_directions)
    
    # Create bigrams from the normalized lemmatized words
    bigrams_directions = [
        tuple(lemmatizer.lemmatize(word) for word in bigram)
        for bigram in ngrams(normalized_directions, 2)]

    # Number of words in directions
    num_words_directions = len(words_in_directions)
    
    # POS tagging for parts of speech analysis
    pos_tags = pos_tag(words_in_directions)

    # Count the occurrences of each part of speech
    pos_counts = Counter(tag for word, tag in pos_tags)

    # Number of steps (approximated by counting verbs 'VB')
    num_steps = len([word for word, tag in pos_tags if tag == 'VB'])
    
    return num_words_title, bigrams_title, polarity, subjectivity,  num_ingredients, num_words_directions, bigrams_directions, pos_counts, num_steps


def add_features(df):
    # Example recipe data
    num_words_title = []
    bigrams_title = []
    polarity = []
    subjectivity = []
    num_ingredients = []
    num_words_directions = []
    bigrams_directions = []
    pos_counts = []
    num_steps = []

    for index, row in df.iterrows():
        title = row['title']
        ingredients = row['ner']
        ingredients_list = json.loads(ingredients)
        directions = row['directions']
        directions_list = json.loads(directions)
        # Analyze the recipe
        #analysis_result = analyze_recipe(ingredients, directions)
        num_title, bi_title, pol, sub, num_ing, num_wd, bi_dir, pos, steps = analyze_recipe(title, ingredients_list, directions_list)
        polarity.append(pol)
        subjectivity.append(sub)
        num_words_title.append(num_title)
        bigrams_title.append(bi_title)
        num_ingredients.append(num_ing)
        num_words_directions.append(num_wd)
        bigrams_directions.append(bi_dir)
        pos_counts.append(pos)
        num_steps.append(steps)
        
    df['num_words_title'] = num_words_title
    df['bigrams_title'] = bigrams_title
    df['subjectivity'] = subjectivity
    df['polarity'] = polarity
    df['num_ingredients'] = num_ingredients
    df['num_words_directions'] = num_words_directions
    df['bigrams_directions'] = bigrams_directions
    df['pos_counts'] = pos_counts
    df['num_steps'] = num_steps
    return df


def extract_info(row):
    try:
        # Extract 'prep_time', 'cook_time', 'total_time', 'servings', and 'yield' dynamically
        prep_time_key = next((key for key in row['prep_data'] if 'prep_time' in key), None)
        cook_time_key = next((key for key in row['prep_data'] if 'cook_time' in key), None)
        total_time_key = next((key for key in row['prep_data'] if 'total_time' in key), None)
        servings_key = next((key for key in row['prep_data'] if 'servings' in key), None)

        # Extract nutrition information if available        
        calories = row['nutritions'].get('calories')
        fat = row['nutritions'].get('fat')
        carbs = row['nutritions'].get('carbs')
        protein = row['nutritions'].get('protein')

        return pd.Series({
            'state': row['state'],
            'title': row['basic_info']['title'],
            'ingredients': row['ingridients'],
            'category': row['basic_info']['category'],
            'rating': row['basic_info']['rating'], 
            'reviews': row['basic_info']['reviews'], 
            'recipe creator': row['basic_info']['recipe_by'],
            'prep_time': row['prep_data'][prep_time_key] if prep_time_key else None,
            'cook_time': row['prep_data'][cook_time_key] if cook_time_key else None,
            'total_time': row['prep_data'][total_time_key] if total_time_key else None,
            'servings': row['prep_data'][servings_key] if servings_key else None,
            'calories': calories,
            'fat': fat,
            'carbs': carbs,
            'protein': protein
        })
    except Exception as e:
        print("Error:", e)
        return pd.Series({})

    
def convert_to_minutes(time_string):
    if pd.isnull(time_string) or time_string.strip() == '':  # Handle missing values and empty strings
        return None
    
    time_string = time_string.lower().replace('.', '')  # Normalize the string by removing periods and converting to lowercase
    time_units = time_string.split()
    
    total_minutes = 0
    days = 0
    hours = 0
    minutes = 0
    
    for idx, unit in enumerate(time_units):
        if 'day' in unit or 'days' in unit:
            days = int(time_units[idx - 1]) 
        if 'hrs' in unit:
            hours = int(time_units[idx - 1])
        if 'mins' in unit:
            minutes = int(time_units[idx - 1])
    total_minutes = (days * 24 * 60) + (hours * 60) + minutes
    return total_minutes

def clean_ingredient(ingredient):
    # This function removes numbers and extra spaces from the ingredient
    return ' '.join([part for part in ingredient.split() if not part.isdigit()])

def find_recipes_with_ingredients(ingredients_to_search, dataframe):
    # Clean the ingredients in the dataframe
    no_null_df['cleaned_ingredients'] = no_null_df['ingredients'].apply(lambda x: [clean_ingredient(ingredient) for ingredient in x])

    matching_recipes = no_null_df[no_null_df['cleaned_ingredients'].apply(
        lambda x: all(ingredient.lower() in ' '.join(x).lower() for ingredient in ingredients_to_search)
    )]

    return matching_recipes

# import transformers
import streamlit as st
import kagglehub
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

# Loading Hugging face token
try:
    load_dotenv()  # Load environment variables from .env file
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    print("Hugging Face Token:", hf_token)
except:
    print("no hugging face token found")

# loading Kaggle dataset
try:
    path = kagglehub.dataset_download("arpansri/books-summary")
    csv_file_path = f"{path}/books_summary.csv" 
    book_data_df = pd.read_csv(csv_file_path).fillna("")
except:
    print("not able to download data")

# loading hugging face model
# Combine summaries and categories
# book_data_df['combined'] = book_data_df['summaries'] + " Categories =" + book_data_df['categories']

# Load the Sentence Transformer model
@st.cache_resource # Cache the model loading to avoid reloading every time
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_data
def generate_embeddings(df):
    df['embedding'] = df['summaries'].apply(lambda x: model.encode(x))
    return df

book_data_df = generate_embeddings(book_data_df)

# Function to recommend books
def recommend_books(book_title, df, category_weight = 0.2):
    # Find the embedding of the entered book title
    if book_title not in df['book_name'].values:
        return "Book not found in the dataset."
    
    book_idx = df[df['book_name'] == book_title].index[0]
    query_embedding = df.loc[book_idx, 'embedding']
    query_category = df.loc[book_idx, 'categories']
    
    # Calculate similarities
    similarities = []
    for idx, row in df.iterrows():
        # Skip the selected book even if there are duplicates
        if row['book_name'] == book_title:
            continue  # Skip the selected book
        
        similarity = util.cos_sim(query_embedding, row['embedding']).item()
         # Category match bonus
        category_bonus = category_weight if row['categories'] == query_category else 0
        hybrid_score = similarity + category_bonus
        similarities.append((row['book_name'], hybrid_score))
    
    # Sort by similarity score and return top 5
    recommendations = sorted(similarities, key=lambda x: x[1], reverse=True)
     # Extract unique books from the top N recommendations
    recommended_books = []
    seen_books = set()
    for book, score in recommendations:
        if book not in seen_books:
            recommended_books.append(book)
            seen_books.add(book)
        if len(recommended_books) == 5:
            break
    return recommended_books

# Streamlit UI
# Initialize session state
if 'clicked_book' not in st.session_state:
    st.session_state['clicked_book'] = None
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = []

st.markdown("""
    <style>
    div.stButton > button:first-child {
        width: 100%;
        background-color: #000000; /* Black background */
        color: #FFD700; /* Gold text */
        border: 2px solid #FFD700; /* Gold border */
        padding: 0.6em;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child: hover {
        background-color: #FFD700; /* Gold background on hover */
        color: #000000; /* Black text on hover */
    }
    </style>
""", unsafe_allow_html=True)

    
st.title("Book Recommendation System")
st.write("Enter the title of a book to get 5 similar recommendations.")

# Input field
# Dropdown for book selection
book_title = st.selectbox("Select book title", book_data_df['book_name'].values)
# Display category and summary when a book is selected
if book_title:
    selected_book = book_data_df[book_data_df['book_name'] == book_title].iloc[0]
    st.write(f"Details of **'{book_title}'**:")
    st.write(f"**Category**: {selected_book['categories']}")
    st.write(f"**Summary**: {selected_book['summaries']}")
    st.session_state['recommendations'] = []

# Add a slider for category weight
category_weight = st.slider(
    "Select category weight",
    min_value=0.0,  # Minimum weight
    max_value=1.0,  # Maximum weight
    value=0.2,      # Default value
    step=0.05,      # Increment step
    help="Adjust the weight for category-based similarity. Higher values emphasize category similarity more."
)

if st.button("Submit", key="styled-button"):
    if book_title.strip() == "":
        st.warning("Please enter a valid book title.")
    else:
        # Get recommendations
        recommendations = recommend_books(book_title, book_data_df, category_weight)
        st.session_state['recommendations'] = recommendations

# Display recommendations in the specified format
if st.session_state['recommendations']:
    for idx, rec in enumerate(st.session_state['recommendations']):
        selected_rec_book = book_data_df[book_data_df['book_name'] == rec].iloc[0]
        st.write(f"{idx+1}- **{rec}:**")
        st.write(f"    - **Category**: {selected_rec_book['categories']}")
        st.write(f"    - **Summary**: {selected_rec_book['summaries']}")
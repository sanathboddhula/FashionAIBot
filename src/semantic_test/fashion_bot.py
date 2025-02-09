import os
import openai
import pinecone
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Hardcoded Pinecone configurations
PINECONE_ENV = "us-east-1-aws"
PINECONE_INDEX_NAME = "fashionproducts"
PINECONE_INDEX_URL = "https://fashionproducts-zn0fky7.svc.aped-4627-b74a.pinecone.io"

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY


def initialize_services():
    """Initialize OpenAI and Pinecone services securely."""
    if not PINECONE_API_KEY:
        st.error("Pinecone API key is missing. Check environment variables.")
        return None

    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pc.Index(PINECONE_INDEX_NAME, PINECONE_INDEX_URL)
    return index


def generate_query_embedding(query):
    """Generate an embedding for the user's query using OpenAI."""
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']


def search_pinecone(index, query_embedding, top_k=5):
    """Perform a semantic search in Pinecone with the query embedding."""
    if not index:
        return None
    
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return search_results


def format_results_as_stylist_response(search_results, query):
    """Use OpenAI to format search results into a conversational response."""
    if not search_results or "matches" not in search_results:
        return "Sorry, I couldn't find any relevant products."

    products = [
        {
            "name": match['metadata']['name'],
            "category": match['metadata']['category'],
            "price": match['metadata']['price'],
            "description": match['metadata'].get('description', ''),
            "url": match['metadata']['url']
        }
        for match in search_results['matches']
    ]

    product_rows = "\n".join(
        f"| {product['name']} | {product['category']} | {product['price']} | {product['description']} | [Link]({product['url']}) |"
        for product in products
    )

    stylist_prompt = f"""
    You are a personal stylist. The user is looking for fashion recommendations based on the following query: "{query}". 
    Based on the products retrieved from a database, craft a conversational response as a stylist, explaining why these products are great choices.
    Here are the products presented in a tabular format:

    | **Name**                        | **Category** | **Price ($)** | **Description**                            | **URL**                     |
    |----------------------------------|--------------|---------------|--------------------------------------------|-----------------------------|
    {product_rows}

    After presenting the table, provide a friendly and stylish response. Be creative and make it engaging.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a fashion stylist."},
            {"role": "user", "content": stylist_prompt}
        ]
    )

    return response['choices'][0]['message']['content']


def main():
    """Streamlit UI for the Fashion AI Chatbot."""
    st.title("Fashion AI Chatbot")
    st.write("Ask for fashion recommendations and let the AI stylist help you find the perfect items!")

    query = st.text_input("What are you looking for?", placeholder="e.g., comfortable black shirts")

    if st.button("Get Recommendations"):
        if query:
            with st.spinner("Fetching recommendations..."):
                index = initialize_services()
                if not index:
                    st.error("Failed to initialize Pinecone services.")
                    return

                query_embedding = generate_query_embedding(query)
                search_results = search_pinecone(index, query_embedding)

                if not search_results or not search_results.get('matches'):
                    st.warning("No results found for your query.")
                    return

                stylist_response = format_results_as_stylist_response(search_results, query)

            st.subheader("Your Personal Stylist Recommends:")
            st.markdown(stylist_response, unsafe_allow_html=True)
        else:
            st.error("Please enter a query to get recommendations.")


if __name__ == "__main__":
    main()

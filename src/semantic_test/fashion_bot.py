import openai
import pinecone
import streamlit as st
from pinecone import Pinecone


def initialize_services():
    """Initialize OpenAI and Pinecone services."""
    openai.api_key = ""
    pc = Pinecone(api_key="",
                  environment="us-east-1-aws")
    index_name = "fashionproducts"
    index = pc.Index(index_name, "https://fashionproducts-zn0fky7.svc.aped-4627-b74a.pinecone.io")
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
    search_results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return search_results


def format_results_as_stylist_response(search_results, query):
    """Use OpenAI to format search results into a conversational response."""
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

    # Create the rows for the table
    product_rows = "\n".join(
        f"| {product['name']} | {product['category']} | {product['price']} | {product['description']} | [Link]({product['url']}) |"
        for product in products
    )

    # Construct the stylist prompt
    stylist_prompt = f"""
    You are a personal stylist. The user is looking for fashion recommendations based on the following query: "{query}". 
    Based on the products retrieved from a database, craft a conversational response as a stylist, explaining why these products are great choices.
    Here are the products presented in a tabular format:

    | **Name**                        | **Category** | **Price ($)** | **Description**                            | **URL**                     |
    |----------------------------------|--------------|---------------|--------------------------------------------|-----------------------------|
    {product_rows}

    After presenting the table, Provide a friendly and stylish response. Be creative and make it engaging.
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

    # User Input
    query = st.text_input("What are you looking for?", placeholder="e.g., comfortable black shirts")

    if st.button("Get Recommendations"):
        if query:
            # Initialize services
            with st.spinner("Fetching recommendations..."):
                index = initialize_services()

                # Generate query embedding
                query_embedding = generate_query_embedding(query)

                # Search Pinecone
                search_results = search_pinecone(index, query_embedding)

                # Format results into stylist response
                stylist_response = format_results_as_stylist_response(search_results, query)

            # Display the recommendations
            st.subheader("Your Personal Stylist Recommends:")
            st.markdown(stylist_response, unsafe_allow_html=True)
        else:
            st.error("Please enter a query to get recommendations.")


if __name__ == "__main__":
    main()

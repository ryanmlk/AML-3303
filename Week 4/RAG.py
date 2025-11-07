# Create a small FAQ dataset for campus information
faq_text = """
Q: When does the library open?
A: The library opens at 8 AM and closes at 8 PM.

Q: Where is the student cafeteria located?
A: The student cafeteria is located on the second floor of the Student Union building.

Q: What are the office hours for academic advising?
A: Academic advising office hours are Monday through Friday, 9 AM to 5 PM.

Q: How do I access the campus WiFi?
A: Connect to "CampusWiFi" network and use your student ID and password to log in.

Q: Where can I find parking on campus?
A: Student parking is available in lots A, B, and C. Lot A is closest to the academic buildings.

Q: When is the registration deadline for next semester?
A: Registration deadline for next semester is November 15th for spring and April 15th for fall.
"""
lines = [chunk.strip() for chunk in faq_text.split("\n\n") if chunk.strip()]

# Create Embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(lines)

# Build FAISS Index
import faiss
import numpy as np

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Example Query
user_question = "What time does the library open?"
q_emb = model.encode([user_question])
D, I = index.search(np.array(q_emb), k=1)
print("Answer:", lines[I[0][0]])

# Streamlit app
import streamlit as st

st.title("Campus FAQ Chatbot")
user_question = st.text_input("Ask your question:")
if user_question:
    q_emb = model.encode([user_question])
    D, I = index.search(np.array(q_emb), k=1)
    st.write("Answer:", lines[I[0][0]])


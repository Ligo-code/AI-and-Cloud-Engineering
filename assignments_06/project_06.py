# --- Step 1: Setup ---

from pathlib import Path
from dotenv import load_dotenv
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY was not found. Check your .env file."

docs_dir = Path("resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

print(f"Document directory found: {docs_dir}")

# --- Step 2: Load Documents ---

documents = SimpleDirectoryReader(str(docs_dir)).load_data()

print(f"\nLoaded {len(documents)} documents:")

for doc in documents:
    print(doc.metadata["file_name"])

# --- Step 3: Build Index and Query Engine ---

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding()
)

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("\nIndex built successfully. Ready to answer questions.")

# --- Step 4: Query the Assistant ---

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\n" + "=" * 80)
    print(f"Question: {question}")

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    top_source = response.source_nodes[0]
    document_name = top_source.node.metadata.get("file_name", "Unknown")
    score = top_source.score
    text_preview = " ".join(top_source.node.get_content().split())[:200]

    print("\nTop retrieved source node:")
    print(f"Document: {document_name}")
    print(f"Score: {score}")
    print(f"Text preview: {text_preview}")

"""
Example output summary:

Q1 Weekend hours:
Answer: Groundwork's hours on weekends are 8:00 AM to 5:00 PM.
Top source: our_story.txt

Q2 Dairy-free milk options:
Answer: All dairy-free options are available at no extra charge.
Top source: seasonal_specials.txt

Q3 Loyalty program:
Answer: You earn one point for every dollar spent. At 100 points, you can redeem a free drink.
Top source: faq.txt

Q4 Company history:
Answer: Groundwork was founded in 2018 by Maya Torres and Sam Okafor in Asheville, North Carolina.
Top source: our_story.txt

Q5 Catering and wholesale:
Answer: Yes, catering and wholesale orders are available.
Top source: wholesale_catering.txt
"""

# --- Reflection on Assistant Responses ---
#
# Overall, the assistant performed reasonably well and answered all five questions correctly,
# but the results revealed several interesting limitations of RAG systems.
#
# Retrieval quality was not always ideal. For the weekend hours question, the assistant gave
# the correct answer (8:00 AM to 5:00 PM), but the top retrieved document was our_story.txt
# instead of faq.txt, where the hours are actually listed. This suggests that the correct
# information was present somewhere in the retrieved context, but ranking was imperfect.
#
# A similar issue appeared in the dairy-free milk question. The assistant correctly stated
# that dairy-free options are available, but did not list the actual milk choices
# (oat, almond, and soy). The top retrieved source was seasonal_specials.txt instead of
# menu.txt, again showing imperfect retrieval relevance.
#
# Some responses were excellent. The loyalty program and company history questions returned
# accurate, detailed answers with clearly relevant source documents.
#
# The catering/wholesale question showed a different limitation. Retrieval worked correctly
# (wholesale_catering.txt was the top result), but the generated answer was too brief and
# omitted useful details such as minimum order requirements, advance notice, and contact email.
# This suggests a generation limitation rather than a retrieval problem.
#
# Overall, this project showed that RAG can significantly improve answer accuracy, but correct
# retrieval ranking does not always happen, and even with good retrieval, the LLM may still
# produce incomplete responses.

# --- Step 5: Find a Failure ---

failure_question = "Which Groundwork location has the fastest Wi-Fi?"

print("\n" + "=" * 80)
print("FAILURE TEST")
print(f"Question: {failure_question}")

response = query_engine.query(failure_question)

print("\nAnswer:")
print(response)

print("\nRetrieved source nodes:")

for i, source_node in enumerate(response.source_nodes, start=1):
    document_name = source_node.node.metadata.get("file_name", "Unknown")
    score = source_node.score
    text_preview = " ".join(source_node.node.get_content().split())[:200]

    print(f"\nSource {i}")
    print(f"Document: {document_name}")
    print(f"Score: {score}")
    print(f"Text preview: {text_preview}")


'''
Output summary for
FAILURE TEST

Question: Which Groundwork location has the fastest Wi-Fi?

Answer:
All Groundwork locations offer free Wi-Fi, so the Wi-Fi speed is likely to be consistent across all three locations.

Retrieved source nodes:

Source 1
Document: our_story.txt
Score: 0.7917795711500055
Text preview: Our Story Groundwork Coffee Co. was founded in 2018 by two college friends, Maya Torres and Sam Okafor, in Asheville, North Carolina. Maya had spent two years working on a coffee farm in Guatemala. Sa

Source 2
Document: faq.txt
Score: 0.77432006791203
Text preview: Frequently Asked Questions Hours - Monday through Friday: 7:00 AM to 7:00 PM - Saturday and Sunday: 8:00 AM to 5:00 PM - We are closed on Thanksgiving Day and Christmas Day. Locations - Downtown: 42 L

Source 3
Document: wholesale_catering.txt
Score: 0.7712112980309181
Text preview: Wholesale and Catering Wholesale Coffee We sell our house blends and single-origin beans in bulk to local restaurants, offices, and retailers. Wholesale pricing is available for orders of 5 pounds or 
'''

# Step 5 Failure Reflection:
# I asked which Groundwork location has the fastest Wi-Fi because the documents mention
# that free Wi-Fi is available at all locations, but they do not provide any information
# about Wi-Fi speed.

# The retrieval was imperfect. The relevant information was in faq.txt, but it was not
# the top result. The top retrieved document was our_story.txt, which is not related to Wi-Fi.

# The model did not invent a specific location or speed, which is good. However, it still made
# an unsupported assumption by saying the speed is "likely to be consistent" across locations.
# That detail is not stated in the documents.

# The model's tone stayed fairly confident even though the retrieved context did not fully
# answer the question. This shows why AI-generated responses should be checked against sources,
# especially when the answer includes words like "likely" or makes an inference from incomplete data.

# To improve the system, I would add stricter instructions telling the model to say when the
# documents do not contain enough information. I would also consider improving retrieval by using
# better chunking, checking all source nodes before answering, or adding metadata filters for FAQs.

"""
Step 6 Final Reflection:

1. Framework Value
In the lesson, building semantic RAG manually required many separate steps, including
document loading, chunking, embedding generation, vector indexing, retrieval logic,
and response generation.

Using LlamaIndex, the equivalent implementation took only a few lines of code:
loading documents with SimpleDirectoryReader, building a VectorStoreIndex,
and creating a query engine.

This shows the value of frameworks in reducing boilerplate code, simplifying implementation,
and making it easier to build production-style AI systems quickly and reliably.

2. Another Business Use Case
A strong real-world use case would be an internal HR assistant for a company.
Employees could ask questions about vacation policy, benefits, onboarding,
expense reimbursement, parental leave, or company policies without manually searching
through multiple HR documents.

This would save time, improve consistency, and reduce repetitive questions for HR teams.

3. Failure Mode RAG Cannot Fully Prevent
RAG improves factual grounding, but it cannot fully prevent incorrect reasoning or unsupported
assumptions by the LLM.

Even when retrieval works correctly, the model may still misinterpret the retrieved information,
combine facts incorrectly, or make confident inferences that are not explicitly supported
by the source documents.

The Wi-Fi failure test demonstrated this: the model correctly retrieved related information
but still made an unsupported assumption about Wi-Fi speed consistency.
"""
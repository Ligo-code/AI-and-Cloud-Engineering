from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- RAG Concepts ---
# Concepts Question 1

"""
Scenario A:
Best approach: RAG
Reason:
The legal team's documents are internal and updated regularly every quarter.
RAG is the best fit because it retrieves the latest information from external documents
at query time without retraining the model every time the policies change.

Scenario B:
Best approach: Fine-tuning
Reason:
The startup wants a very specific writing style (brand voice) that the model should learn
as behavior, not retrieve as facts. Fine-tuning is appropriate because they already have
3,000 examples of high-quality internal writing.

Scenario C:
Best approach: Prompt Engineering (Context Injection)
Reason:
This is just a single short report (two pages), so building a full retrieval system
would be unnecessary overhead. Simply putting the report content into the prompt is the
simplest and cheapest solution.
"""

# Concepts Question 2

"""
A confidently wrong answer is more harmful because people are more likely
to trust information when it sounds certain and convincing.

When an answer looks believable, users may not question it or verify it elsewhere.
This can lead to serious consequences because they may act on incorrect information.

For example, if someone asks whether they need a visa to travel to a certain country
and the AI confidently says no when a visa is actually required, the person could
book a trip, arrive at the airport, and be denied entry.

Another example is asking whether a mushroom is safe to eat.
If the AI incorrectly identifies a poisonous mushroom as edible, someone could get seriously sick.

If the model instead said "I am not sure," the user would be much more likely
to double-check the information before taking action.

The tone matters because confidence increases trust, even when the information is wrong.
"""

# Concepts Question 3

"""
Correct RAG pipeline order:

# Preprocessing and Indexing (done once when setting up the system) offline in advance

1. Extract text from source documents
   Extract raw text from PDFs or other source files.

2. Split text into chunks
   Break large documents into smaller manageable pieces.

3. Convert text chunks into embeddings
   Transform each chunk into vector representations for semantic search.

# Query Time (done every time a user asks a question) online in real-time

4. Receive the user's query
   Accept the user's question.

5. Embed the user's query
   Convert the query into a vector using the same embedding model.

6. Retrieve the most relevant chunks
   Compare vectors and find the most similar document chunks.

7. Inject retrieved chunks into the prompt
   Add the retrieved context to the prompt sent to the LLM.

8. Generate a response from the LLM
   The model produces an answer based on the retrieved context.
"""

# --- Keyword RAG ---

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]
    
# Keyword Question 1

query = "What are your hours on the weekend?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nKeyword Q1 selected document:")
print(results[0][0])

"""
Keyword Question 1 Explanation:
The retrieval selected loyalty.txt instead of hours.txt.

This happened because the keyword matching relies on exact word overlap.
The query included words like "your" that matched unrelated documents,
while "weekend" did not match "weekends" because keyword matching does not handle word variations.

This shows a limitation of simple keyword-based retrieval.

"""

# Keyword Question 2

query = "Do you have anything without caffeine?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nKeyword Q2 selected document:")
print(results[0][0])

"""
Keyword Question 2 Explanation:
No document was selected because there were no exact overlapping keywords
between the query and the document contents.

Keyword retrieval failed here because it only matches exact words, not meaning.
The menu document is actually relevant because it lists coffee drinks, which contain caffeine,
but the word "caffeine" itself does not appear.

Semantic retrieval would work better because it understands conceptual similarity.
It could recognize that "espresso" and "cappuccino" are related to caffeine,
even without exact word overlap.
"""

# Keyword Question 3

query = "How do I sign up for rewards?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

"""
Prediction:
I think the selected document will be "loyalty.txt".

Reason:
"free drink" is a strong keyword match for "rewards" in the query, and "loyalty program" is also closely related to "rewards".
"""

results = simple_keyword_retrieval(query, documents, verbose=True)

print("\nKeyword Q3 selected document:")
print(results[0][0])

"""
Was my prediction correct?
No, my prediction was not correct. I expected loyalty.txt, but the function returned "None found".

Explanation:
As a human, I understood that "rewards" is related to a loyalty program, points, and free drinks.
However, simple keyword retrieval only checks exact word overlap.
The document does not contain the exact word "rewards" or the phrase "sign up",
so the function could not find a match.
"""

# --- Semantic RAG Concepts ---
# Semantic Question 1

"""
What is a vector embedding?
A vector embedding is a numerical representation of text that captures its meaning.
Texts with similar meanings will have vectors that are close to each other in vector space.

Which chunk is more relevant: cosine similarity 0.85 or 0.30?
The chunk with a similarity score of 0.85 is more relevant because a higher cosine similarity
means the meanings of the query and the text chunk are more closely aligned.

Why can semantic search find relevant chunks without exact matching words?
Semantic search compares meaning rather than exact words.
For example, a query about "cars" may still match a document containing the word "automobile"
because their meanings are similar, even though the words are different.
"""

# Semantic Question 2

"""
| Feature                    | Keyword RAG                       | Semantic RAG                     |
|----------------------------|-----------------------------------|----------------------------------|
| What is compared?          | Exact word overlap                | Vector similarity (meaning)      |
| What is retrieved?         | Full document                     | Relevant text chunks             |
| Can it handle synonyms?    | No                                | Yes                              |
| Storage format             | Plain text dictionary             | Vector database / vector index   |
| Relevance score            | Number of overlapping keywords    | Cosine similarity score          |
"""

# --- LlamaIndex ---

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PDFReader

# LlamaIndex Question 1

documents = SimpleDirectoryReader(
    "resources/brightleaf_pdfs",
    file_extractor={".pdf": PDFReader()}
).load_data()

print(f"\nLoaded {len(documents)} documents.")

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding()
)

print("Vector index created successfully.")

query_engine = index.as_query_engine(
    similarity_top_k=3
)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for question in questions:
    print("\n" + "=" * 80)
    print(f"Question: {question}")

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    print("\nRetrieved source nodes:")
    for i, source_node in enumerate(response.source_nodes, start=1):
        score = source_node.score
        text_preview = source_node.node.get_content().replace("\n", " ")[:150]

        print(f"\nSource {i}")
        print(f"Score: {score}")
        print(f"Text preview: {text_preview}")

'''
Output:
Loaded 6 documents.
2026-05-20 22:14:06,202 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
Vector index created successfully.

================================================================================
Question: What employee benefits does BrightLeaf offer?
2026-05-20 22:14:06,655 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2026-05-20 22:14:08,124 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

Answer:
BrightLeaf offers a comprehensive benefits program that includes health benefits such as medical insurance, vision benefits, 
and wellness programs. Additionally, they provide financial security benefits like life insurance, disability insurance, 
and a 401(k) retirement plan with a company match. Employees also receive parental leave, work flexibility options, 
professional development stipends, mentorship programs, and access to free online courses. The company emphasizes diversity, 
equity, and inclusion through its policies and practices.

Retrieved source nodes:

Source 1
Score: 0.9130897402885385
Text preview: Introduction BrightLeaf Solar views employee well-being as inseparable from long-term innovation. 
Our benefits program is designed to help each team m

Source 2
Score: 0.8221282838314069
Text preview: Network and Data Security BrightLeaf maintains layered defenses for all production and corporate networks. 
Access to critical systems requires multi■f

Source 3
Score: 0.8188774838837097
Text preview: Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. 
Our mission is to make solar power pract

================================================================================
Question: What are BrightLeaf's security policies?
2026-05-20 22:14:08,282 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2026-05-20 22:14:09,801 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

Answer:
BrightLeaf's security policies include maintaining layered defenses for all networks, 
requiring multi-factor authentication and VPN with device certificates for access to critical systems, 
rotating credentials every 90 days, encrypting customer data in transit and at rest with keys stored in a managed HSM, 
enforcing least privilege through perimeter firewalls and cloud security groups, centralizing logs with anomaly detection, 
following an incident response plan based on NIST 800-61 guidance, conducting tabletop exercises, 
providing employee training on security, conducting quarterly RBAC reviews, 
ensuring vendor and supply chain security through security questionnaires and network segmentation, 
aligning with ISO 27001 practices for compliance, and tracking security metrics and encouraging employees to report suspected issues.

Retrieved source nodes:

Source 1
Score: 0.8891485530002787
Text preview: Network and Data Security BrightLeaf maintains layered defenses for all production and corporate networks. 
Access to critical systems requires multi■f

Source 2
Score: 0.8414286925839102
Text preview: Introduction BrightLeaf Solar views employee well-being as inseparable from long-term innovation. 
Our benefits program is designed to help each team m

Source 3
Score: 0.8239698727127658
Text preview: Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. 
Our mission is to make solar power pract
'''

"""
LlamaIndex Question 1 Observations:

Query 1: What employee benefits does BrightLeaf offer?
The retrieved chunks look mostly relevant. The first chunk is clearly about employee benefits,
while the second and third chunks are less directly relevant.
The model's response sounds confident and specific because it lists concrete benefits such as
medical insurance, vision benefits, wellness programs, life insurance, disability insurance,
401(k), parental leave, flexibility, and professional development.
It was a little unexpected that security and company overview chunks were also retrieved.

Query 2: What are BrightLeaf's security policies?
The retrieved chunks look mostly relevant. The first chunk is clearly about network and data security,
while the second and third chunks are less directly relevant.
The model's response sounds confident and specific because it includes details about MFA, VPN,
credential rotation, encryption, least privilege, logs, incident response, RBAC reviews,
vendor security, and ISO 27001 practices.
It was unexpected that the benefits and company overview chunks were also retrieved.
"""

# LlamaIndex Question 2

question = "What employee benefits does BrightLeaf offer?"

for top_k in [1, 5]:
    print("\n" + "=" * 80)
    print(f"LlamaIndex Q2 | similarity_top_k={top_k}")
    print(f"Question: {question}")

    query_engine = index.as_query_engine(
        similarity_top_k=top_k
    )

    response = query_engine.query(question)

    print("\nAnswer:")
    print(response)

    print("\nRetrieved source nodes:")
    for i, source_node in enumerate(response.source_nodes, start=1):
        score = source_node.score
        text_preview = source_node.node.get_content().replace("\n", " ")[:150]

        print(f"\nSource {i}")
        print(f"Score: {score}")
        print(f"Text preview: {text_preview}")

'''
Output:
================================================================================
LlamaIndex Q2 | similarity_top_k=1
Question: What employee benefits does BrightLeaf offer?
2026-05-20 22:18:59,737 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2026-05-20 22:19:01,706 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

Answer:
BrightLeaf Solar offers a comprehensive benefits program that includes health, vision, and wellness benefits such as medical insurance, 
vision benefits, wellness programs, and a Wellness Reimbursement Plan. Additionally, the company provides financial security 
and retirement benefits like life insurance, disability insurance, and a 401(k) retirement plan with a company match. 
BrightLeaf also offers parental leave, work flexibility, professional development stipends, internal mentorship programs, 
and resources for diversity, equity, and inclusion.

Retrieved source nodes:

Source 1
Score: 0.9130897402885385
Text preview: Introduction BrightLeaf Solar views employee well-being as inseparable from long-term innovation. 
Our benefits program is designed to help each team m

================================================================================
LlamaIndex Q2 | similarity_top_k=5
Question: What employee benefits does BrightLeaf offer?
2026-05-20 22:19:02,693 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2026-05-20 22:19:04,026 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

Answer:
BrightLeaf offers a comprehensive benefits program that includes health benefits such as medical insurance, vision benefits, 
and wellness programs. They also provide financial security benefits like life insurance, disability insurance, 
and a 401(k) retirement plan with a company match. Additionally, BrightLeaf offers parental leave, work flexibility, 
professional development opportunities, mentorship programs, and diversity and inclusion initiatives to support employee growth 
and well-being.

Retrieved source nodes:

Source 1
Score: 0.9130897402885385
Text preview: Introduction BrightLeaf Solar views employee well-being as inseparable from long-term innovation. 
Our benefits program is designed to help each team m

Source 2
Score: 0.8221282838314069
Text preview: Network and Data Security BrightLeaf maintains layered defenses for all production and corporate networks. 
Access to critical systems requires multi■f

Source 3
Score: 0.8188774838837097
Text preview: Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. 
Our mission is to make solar power pract

Source 4
Score: 0.813724315119906
Text preview: EcoVolt Energy (2022 Partnership) BrightLeaf's collaboration with EcoVolt Energy, established in 2022, 
focused on delivering microgrid solutions to ru

Source 5
Score: 0.7911937121114855
Text preview: Overview This report summarizes BrightLeaf Solar's financial performance from 2021 through 2025. 
The period includes a growth phase, a temporary dip i
'''

"""
LlamaIndex Question 2 Observation:
With similarity_top_k=1, the response was already strong because the top retrieved chunk
was directly about employee benefits.

With similarity_top_k=5, the answer did not improve much. The response was similar,
but more unrelated chunks were retrieved, including security, company overview,
partnerships, and financial performance.

This shows that more retrieved context is not always better.
If the additional chunks are not relevant, they can add noise and potentially make
the model's answer less focused.
"""

# LlamaIndex Question 3

question = "What is BrightLeaf's vacation policy for pregnant contractors?"

print("\n" + "=" * 80)
print("LlamaIndex Question 3")
print(f"Question: {question}")

query_engine = index.as_query_engine(
    similarity_top_k=3
)

response = query_engine.query(question)

print("\nAnswer:")
print(response)

print("\nRetrieved source nodes:")
for i, source_node in enumerate(response.source_nodes, start=1):
    score = source_node.score
    text_preview = source_node.node.get_content().replace("\n", " ")[:150]

    print(f"\nSource {i}")
    print(f"Score: {score}")
    print(f"Text preview: {text_preview}")

'''
Output:
================================================================================
LlamaIndex Question 3
Question: What is BrightLeaf's vacation policy for pregnant contractors?
2026-05-20 22:53:12,831 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2026-05-20 22:53:13,995 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

Answer:
All new parents, regardless of gender, marital status, or family structure, receive twelve weeks of paid leave. 
Additional unpaid leave can be arranged as needed.

Retrieved source nodes:

Source 1
Score: 0.8358035606436087
Text preview: Introduction BrightLeaf Solar views employee well-being as inseparable from long-term innovation. 
Our benefits program is designed to help each team m

Source 2
Score: 0.7771875581366439
Text preview: Network and Data Security BrightLeaf maintains layered defenses for all production and corporate networks. 
Access to critical systems requires multi■f

Source 3
Score: 0.7742292247398805
Text preview: Overview BrightLeaf Solar was founded on the belief that renewable energy should be a right, not a privilege. 
Our mission is to make solar power pract
'''

'''
LlamaIndex Question 3 Observation:
I expected this query to be difficult because it asks about a very specific group:
pregnant contractors.

The system retrieved the benefits document and answered with the general parental leave policy:
all new parents receive twelve weeks of paid leave, with additional unpaid leave available.

However, the answer does not clearly address whether the policy applies to contractors.
This is a limitation because the model gave a confident answer based on related context,
but it did not explain that the retrieved text may not contain enough information about contractors.

To improve the system, I would add stricter instructions to say when the answer is not fully
supported by the context. I would also include better metadata or filters for employee type
if the documents contain separate policies for employees and contractors.

'''

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# LlamaIndex Question 4

judge_llm = OpenAI(model="gpt-4o-mini")

faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

evaluation_questions = [
    "What employee benefits does BrightLeaf offer?",
    "What is BrightLeaf's policy for company-owned submarines?",
]

query_engine = index.as_query_engine(similarity_top_k=3)

for q in evaluation_questions:
    print("\n" + "=" * 80)
    print(f"Evaluation question: {q}")

    response = query_engine.query(q)

    faithfulness_result = faithfulness_evaluator.evaluate_response(
        query=q,
        response=response
    )

    relevancy_result = relevancy_evaluator.evaluate_response(
        query=q,
        response=response
    )

    print("\nResponse:")
    print(response)

    print("\nFaithfulness score:")
    print(faithfulness_result.score)

    print("\nRelevancy score:")
    print(relevancy_result.score)

    '''
    Output:
    ================================================================================
Evaluation question: What employee benefits does BrightLeaf offer?
2026-05-20 23:00:23,580 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2026-05-20 23:00:24,914 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-05-20 23:00:25,933 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-05-20 23:00:26,006 - INFO - Retrying request to /chat/completions in 0.496313 seconds
2026-05-20 23:00:27,283 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

Response:
BrightLeaf offers a comprehensive benefits program that includes health benefits such as medical insurance, vision benefits, and wellness programs. They also provide financial security through life, disability, and retirement benefits. Additionally, BrightLeaf offers parental leave, work flexibility, professional development opportunities, mentorship programs, and access to free online courses. The company emphasizes diversity, equity, and inclusion through its policies and practices to support the well-being and growth of its employees.

Faithfulness score:
1.0

Relevancy score:
1.0

================================================================================
Evaluation question: What is BrightLeaf's policy for company-owned submarines?
2026-05-20 23:00:27,495 - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2026-05-20 23:00:28,247 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-05-20 23:00:28,268 - INFO - Retrying request to /chat/completions in 0.400883 seconds
2026-05-20 23:00:29,449 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-05-20 23:00:29,473 - INFO - Retrying request to /chat/completions in 0.499928 seconds
2026-05-20 23:00:30,728 - INFO - HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

Response:
There is no mention of BrightLeaf having a policy for company-owned submarines in the provided context information.

Faithfulness score:
0.0

Relevancy score:
0.0
    '''

'''
"""
LlamaIndex Question 4 Observation:

A faithfulness score of 1.0 means the answer is supported by the retrieved context.
A faithfulness score of 0.0 means the answer is not supported by the retrieved context
or the evaluator does not consider it grounded in the provided sources.

A relevancy score measures whether the response actually answers the user's question.
This is different from faithfulness because an answer can be grounded in the context
but still fail to answer the question directly.

The scores changed between the two queries.
For the employee benefits question, both scores were 1.0 because the documents contained
relevant information and the answer was supported by the retrieved context.
For the submarine question, both scores were 0.0 because the documents did not contain
information about company-owned submarines.

The LLM-as-a-judge approach means using another LLM to evaluate the quality of a generated answer.
It is useful for RAG evaluation because many answers are open-ended and cannot be checked with
a simple exact-match accuracy metric.
"""
'''
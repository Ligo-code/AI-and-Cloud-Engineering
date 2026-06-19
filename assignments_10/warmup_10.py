# --- LLMs as Transform ---

# Q1
# 1. Deterministic code: date parsing has a clear correct output and can be handled reliably with Python.
# 2. LLM: classifying a freeform support ticket requires language understanding and judgment.
# 3. Deterministic code: calculating an average is arithmetic and should be done with Python.
# 4. LLM: extracting a company name from messy freeform text can be ambiguous and irregular.
# 5. Deterministic code: counting words is simple, reliable, and does not need an LLM.

# Q2
# Problem:
# The prompt "Summarize this product review in a few sentences" creates unpredictable output.
# In a pipeline, that is hard to parse, store consistently, or validate because the model may return
# different lengths, formats, or extra commentary.
#
# Better prompt:
# "Summarize this product review in exactly one sentence. Reply with the summary text only,
# with no bullet points, no labels, and no extra explanation."

# Q3
# 50,000 records * 1 second per record = 50,000 seconds.
# 50,000 seconds is about 833 minutes, or about 13.9 hours, if processed sequentially.
#
# One practical strategy is to process records in batches or use concurrent/asynchronous requests
# so multiple records are classified at the same time without changing the model.

# --- Azure OpenAI ---

# Q1
# An organization might use Azure OpenAI for data residency and compliance, especially if it already
# runs regulated workloads in Azure and needs requests to stay within Azure infrastructure.
# Another reason is unified billing, procurement, and support through Microsoft instead of managing
# a separate vendor relationship with OpenAI directly.

# Q2
# azure_endpoint: the URL of the Azure OpenAI resource, such as
# "https://<resource-name>.openai.azure.com".
# api_version: the Azure OpenAI API version used by the client, such as "2024-02-01".
# azure_deployment: the named deployment created in Azure for a specific model.
# Note: in the lesson example, the deployment name is passed later as the model value.

# Q3
# With AzureOpenAI, the model parameter takes the deployment name, not the base model name.
# For example, instead of "gpt-4o-mini", it might be "gpt4o-mini-prod".
# The correct deployment name is found in Azure AI Foundry under the Azure OpenAI resource's
# Deployments section, or it is provided by the platform/infrastructure team.
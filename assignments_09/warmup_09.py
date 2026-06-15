# --- Azure Authentication ---

# Q1
# When running a Python script locally with DefaultAzureCredential,
# it relies on the Azure CLI login session from my local machine.
# Before running the script, I need to run:
#
# az login
#
# DefaultAzureCredential tries several authentication methods in order.
# During local development, it can detect that I am already signed in
# through the Azure CLI and use that session automatically.
# This means I do not need to hardcode a username, password, or token
# in my Python script.


# Q2
# A deployed pipeline running on an Azure VM or container cannot use
# az login because there is no human sitting at a terminal to complete
# the login flow.
#
# In production, the pipeline should use a managed identity instead.
# A managed identity is an Azure-managed identity attached to the VM,
# container, or other Azure resource.
#
# The same Python code works without changes because
# DefaultAzureCredential automatically checks for managed identity
# when the code is running in Azure. Locally it can use az login;
# in Azure it can use managed identity.


# Q3
# If a script creates DefaultAzureCredential and immediately gets an
# AuthenticationError, the two most likely causes are:
#
# 1. I am not logged in to Azure locally, or my Azure CLI session expired.
#    I would diagnose this by running:
#
#    az account show
#
#    If that fails, I would run:
#
#    az login
#
# 2. I am logged into the wrong Azure account or subscription, or my account
#    does not have permission to access the Azure resource.
#    I would diagnose this by checking:
#
#    az account show
#    az account list
#
#    and by confirming in the Azure Portal that my account has the correct
#    role/permissions for the subscription or storage account.

# --- Blob Storage ---

# Q1
# Azure Blob Storage has three levels:
#
# 1. Storage Account - the top-level resource that contains all storage data.
# 2. Container - a logical grouping of blobs within a storage account.
# 3. Blob - an individual file stored in a container.
#
# Filing cabinet analogy:
#
# Storage Account = the entire filing cabinet
# Container = a drawer inside the cabinet
# Blob = an individual document inside the drawer
#
# Another analogy:
#
# Storage Account = hard drive
# Container = folder
# Blob = file


# Q2
#
# Scenario 1:
# A REST API returns a JSON payload each hour.
# Store the raw responses in Blob Storage because the data is being
# saved as files and may need to be reprocessed later.
#
# Scenario 2:
# A pipeline produces 50 million customer transactions that analysts
# query by date range and customer ID.
# Use a relational database because the data needs efficient querying,
# filtering, and indexing.
#
# Scenario 3:
# A computer vision model produces image embeddings as NumPy arrays.
# Store them in Blob Storage because they are file-like artifacts that
# need to be saved and reused between pipeline runs.

# Q3
def list_container(container_client):
    """
    Print the name and size of every blob in the container.
    """
    for blob in container_client.list_blobs():
        print(f"{blob.name} {blob.size} bytes")


# Q4
def upload_text(container_client, blob_name, text):
    """
    Encode a Python string as UTF-8 and upload it as a blob.
    """
    data = text.encode("utf-8")
    container_client.upload_blob(blob_name, data, overwrite=True)

# Week 8 Warmup

## Cloud Concepts Question 1

### What is the core economic model of cloud computing, and how does it differ from owning your own servers?

Cloud computing uses a pay-as-you-go model, meaning organizations pay only for the resources they actually use. This differs from owning physical servers, where companies must purchase hardware upfront and cover ongoing costs such as maintenance, power, cooling, and upgrades regardless of actual usage.

---

## Cloud Concepts Question 2

### What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Vertical scaling means increasing the resources of a single machine, such as adding more CPU, RAM, or a faster GPU. Horizontal scaling means adding more machines and distributing the workload across them.

Vertical scaling is useful when an application needs more power on a single machine. Horizontal scaling is useful when a workload can be distributed across multiple machines and demand continues to grow.

#### Scenario 1

**A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.**

Horizontal scaling, because additional servers can be added to distribute the increased traffic.

#### Scenario 2

**A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.**

Vertical scaling, because the existing machine is being upgraded with more powerful hardware.

#### Scenario 3

**A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.**

Horizontal scaling, because the workload can be distributed across multiple machines running in parallel.

---

## Cloud Concepts Question 3

### Classify each item as IaaS, PaaS, or SaaS.

#### Gmail — SaaS

Users access the application through a browser while Google manages the software, platform, and infrastructure.

#### Azure Virtual Machines — IaaS

Azure provides the virtual hardware while the user manages the operating system, applications, and configuration.

#### Azure App Service — PaaS

Azure manages the infrastructure and runtime environment while developers focus on deploying and maintaining their applications.

#### AWS S3 (Simple Storage Service) — PaaS

AWS manages the storage infrastructure while developers focus on storing and retrieving data.

#### GitHub Codespaces — PaaS

GitHub provides a managed development environment, allowing developers to focus on coding rather than infrastructure management.

#### Snowflake — SaaS

Snowflake is a fully managed analytics platform that users access without managing the underlying infrastructure.

### Describe IaaS, PaaS, and SaaS in your own words.

#### IaaS (Infrastructure as a Service)

IaaS provides virtualized computing resources such as servers, storage, and networking. The cloud provider manages the physical infrastructure, while the customer manages the operating system, applications, and data.

**Example:** Azure Virtual Machines

As a developer, I would be responsible for configuring the operating system, installing software, applying updates, and managing my applications.

#### PaaS (Platform as a Service)

PaaS provides a managed platform for building and running applications. The cloud provider manages the infrastructure, operating system, and runtime environment.

**Example:** Azure App Service

As a developer, I would focus on deploying and maintaining application code while Azure manages the underlying platform.

#### SaaS (Software as a Service)

SaaS delivers complete software applications over the internet. The provider manages everything from infrastructure to application updates.

**Example:** Gmail

As a user, I only manage my account and data while Google manages the entire application and infrastructure.

---

## Cloud Concepts Question 4

### What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?

A managed data platform such as Databricks or Snowflake is built on top of cloud providers and is optimized specifically for data and analytics workloads. These platforms simplify setup, scaling, and infrastructure management, making it faster to start working with data. The tradeoff is reduced flexibility and potentially higher costs compared to building and managing your own solutions directly in Azure, AWS, or GCP.

---

## Cloud Concepts Question 5

### The lesson names two situations where the cloud is probably not the right choice. What are they?

1. When a dataset fits comfortably on a single machine and does not require large amounts of compute power, local processing is often faster and cheaper.
2. When the added complexity and learning curve of cloud infrastructure outweigh the benefits for a simple project or prototype.

---

## Azure Basics Question 1

### What is the difference between an Azure subscription and a resource group? Which one is yours alone, and which one does CTD share?

An Azure subscription is the billing and resource-management boundary within Azure. A resource group is a logical container used to organize related resources.

Code the Dream shares a single Azure subscription, while each student has their own personal resource group.

---

## Azure Basics Question 2

### Azure Cloud Shell is ephemeral by default. What does that mean in practice, and what does your course setup use to make it persistent?

An ephemeral environment means that files stored locally in Cloud Shell are deleted when the session ends. In this course, persistence is achieved by mounting Azure Storage through a file share, allowing files in the home directory to remain available across sessions.

---

## Azure Basics Question 3

### What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?

The SSH private key is a secret credential that must remain on the user's machine and should never be shared. The SSH public key can be shared safely and is uploaded to remote systems.

This is safe because the remote system can verify that the user possesses the matching private key without ever receiving the private key itself.

---

## Azure Basics Question 4

Run the following command in Cloud Shell:

```bash
az account show
```

### Output

{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#rigatovt@gmail.com",
    "type": "user"
  }
}

### What changes when you add `--output table`?

The `--output table` option formats the JSON output into a human-readable table, making it easier to read in the terminal.

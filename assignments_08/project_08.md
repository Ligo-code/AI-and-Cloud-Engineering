## Video Link






## Cost Analysis Summary
Scenario A used a Standard_B1s virtual machine running 160 hours per month and cost approximately $1.66 per month.

Scenario B included a Standard_NC6s_v3 GPU virtual machine running continuously, an Azure SQL Database with 4 vCores, 
and 1 TB of Blob Storage. The total estimated monthly cost was approximately $3,000.

The most surprising finding was how dramatically GPU compute increased the cost. While the lightweight VM cost less than $2 per month, 
the GPU-enabled VM alone cost more than $2,200 per month.

The Python script calculated the VM costs using the hourly rates from the Pricing Calculator. 
The results closely matched the values shown in Azure Pricing Calculator, with only minor differences due to rounding.
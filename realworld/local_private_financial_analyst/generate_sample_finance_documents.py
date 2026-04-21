import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_pdf(filename, title, content):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, title)
    
    # Content
    c.setFont("Helvetica", 12)
    y = height - 80
    for line in content:
        c.drawString(50, y, line)
        y -= 20
        # Basic pagination if content is too long
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50
            
    c.save()

# Sample Data Definitions
bank_content = [
    "Date | Description | Amount | Balance",
    "-----------------------------------------",
    "01/01/2026 | Opening Balance | | $12,450.00",
    "01/05/2026 | Deposit - Payroll / TechCorp | +$4,200.00 | $16,650.00",
    "01/12/2026 | Payment - Amazon Web Services | -$154.20 | $16,495.80",
    "01/15/2026 | Transfer - Savings Account | -$1,000.00 | $15,495.80",
    "01/20/2026 | Purchase - Apple Store | -$1,299.00 | $14,196.80",
    "01/28/2026 | Service Fee - Maintenance | -$15.00 | $14,181.80",
    "01/31/2026 | Closing Balance | | $14,181.80"
]

invoice_content = [
    "Invoice No: UDM-2026-9901",
    "Invoice Date: February 15, 2026",
    "Billed To: Eason Wu (123 AI Lane, Tech City)",
    "Issued By: QuantumVolt Labs (support@quantumvolt.ai)",
    "Description: AI Strategy Consulting - Phase 1",
    "Quantity: 1",
    "Unit Price: $2,500.00",
    "Tax (GST 10%): $250.00",
    "Total Amount Due: $2,750.00",
    "Payment Terms: Net 30 days"
]

tax_content = [
    "Tax Year: 2026",
    "Filing Status: Single",
    "Primary SSN: XXX-XX-4285",
    "Total Wages/Salaries: $115,200.00",
    "Taxable Interest: $450.00",
    "Adjusted Gross Income (AGI): $112,400.00",
    "Standard Deduction: $15,000.00",
    "Total Tax Paid: $22,450.00",
    "Refund Amount: $1,150.00"
]

# Ensure the directory exists
target_dir = "private_finance_docs"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Generate the files
create_pdf(f"{target_dir}/bank_statement_q1.pdf", "Bank Statement - Q1 2026", bank_content)
create_pdf(f"{target_dir}/invoice_udm_2026.pdf", "Service Invoice", invoice_content)
create_pdf(f"{target_dir}/tax_transcript_2026.pdf", "Tax Return Transcript 2026", tax_content)

print(f"✅ Generated 3 sample PDFs in the '{target_dir}' folder.")
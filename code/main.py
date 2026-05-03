import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool

# Load environment variables
load_dotenv()

# --- Section 1: Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
TICKETS_PATH = PROJECT_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_PATH = PROJECT_ROOT / "support_tickets" / "output.csv"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure API Key is available
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Model selection
MODEL_NAME = "gemini/gemini-flash-lite-latest"

# --- Section 2: Custom Tools ---
class GroundedSearchTool(BaseTool):
    """
    A tool that performs domain-aware searches within the local /data corpus.
    Ensures that agents only answer based on verified documentation.
    """
    name: str = "Grounded Search Tool"
    description: str = "Search for official company documentation in the /data directory."

    def _run(self, company: str) -> str:
        # Determine folders to search
        folders_to_search = []
        if not company or pd.isna(company) or str(company).lower() == 'none':
            # Search all folders if company is unknown
            folders_to_search = [DATA_DIR / "hackerrank", DATA_DIR / "claude", DATA_DIR / "visa"]
        else:
            folders_to_search = [DATA_DIR / str(company).lower()]
            
        results = []
        for company_path in folders_to_search:
            if not company_path.exists():
                continue
                
            # Scan for markdown and text files
            for file_path in company_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in [".md", ".txt"]:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            results.append(f"--- File: {file_path.name} ({company_path.name}) ---\n{f.read()}\n")
                    except Exception:
                        continue
                        
        # Return truncated results to fit context window
        return "\n".join(results)[:4000] if results else "No documentation found for this query."

# --- Section 3: Agent Definitions ---
# Triage Agent: The Gateway
triage_agent = Agent(
    role="Triage Manager",
    goal="Classify user queries into Safety, Search, or Tech domains.",
    backstory="You are the expert gateway for support. Analyze the ticket and provide the correct domain classification.",
    llm=MODEL_NAME
)

# Specialist Agent: The Knowledge Master
search_agent = Agent(
    role="Search Specialist",
    goal="Answer all FAQ and product queries using grounded documentation.",
    backstory="You are a knowledge retrieval expert. You use the search tool to find exact answers for {company}.",
    tools=[GroundedSearchTool()],
    llm=MODEL_NAME
)

# Supervisor Agent: The Final Auditor
supervisor_agent = Agent(
    role="Support Supervisor",
    goal="Audit the response for 100% truthfulness and format it professionally.",
    backstory=(
        "The final quality controller. You ensure every response is grounded and concise. "
        "You MUST remove any 'Thought:' or internal reasoning from the final response. "
        "Allowed status: 'replied', 'escalated'. "
        "Allowed request_type: 'product_issue', 'feature_request', 'bug', 'invalid'."
    ),
    tools=[GroundedSearchTool()],
    llm=MODEL_NAME
)

# --- Section 4: Task Definitions ---
def process_ticket(row):
    """
    Orchestrates a hierarchical multi-agent crew to resolve a single ticket.
    Includes built-in retry logic for API resilience.
    """
    inputs = {
        "issue": row.get('Issue', ''),
        "subject": row.get('Subject', ''),
        "company": row.get('Company', '')
    }

    triage_task = Task(
        description="Analyze the ticket: Subject: {subject}, Issue: {issue}. Classify as Safety, Search, or Tech.",
        expected_output="A single word domain classification.",
        agent=triage_agent
    )

    specialist_task = Task(
        description="Resolve the issue for {company}. Use Grounded Search Tool to find the official answer. Include evidence.",
        expected_output="A professional draft response based on documentation.",
        agent=search_agent,
        context=[triage_task]
    )

    supervisor_task = Task(
        description=(
            "Review the draft for {company}. Ensure it is concise (max 3-5 sentences). "
            "STRICT RULE: Your output MUST NOT contain internal reasoning or 'Thought:' labels. "
            "Provide: response, status, product_area, request_type, justification."
        ),
        expected_output=(
            "Final Response: [Clean text]\n"
            "Status: [replied/escalated]\n"
            "Product Area: [area]\n"
            "Request Type: [type]\n"
            "Justification: [explanation]"
        ),
        agent=supervisor_agent,
        context=[specialist_task]
    )

    crew = Crew(
        agents=[triage_agent, search_agent, supervisor_agent],
        tasks=[triage_task, specialist_task, supervisor_task],
        process=Process.sequential,
        max_rpm=10
    )

    # Resilience Loop
    for attempt in range(2):
        try:
            return crew.kickoff(inputs=inputs)
        except Exception as e:
            if attempt == 0 and ("429" in str(e) or "503" in str(e)):
                print(f"Server busy. Retrying in 30s...")
                time.sleep(30)
                continue
            return None

# --- Section 5: Batch Processing & Persistence ---
def run_batch_processing():
    """
    Main loop to process tickets and persist results to output.csv.
    """
    if not TICKETS_PATH.exists():
        print(f"Error: {TICKETS_PATH} not found.")
        return

    input_df = pd.read_csv(TICKETS_PATH)
    
    # Load progress for resumability
    if OUTPUT_PATH.exists():
        try:
            output_df = pd.read_csv(OUTPUT_PATH)
            processed_subjects = set(output_df['Subject'].astype(str).tolist())
        except Exception:
            processed_subjects = set()
    else:
        processed_subjects = set()

    print(f"Starting Orchestration. {len(processed_subjects)} tickets already processed.")

    for index, row in input_df.iterrows():
        subject_str = str(row.get('Subject', 'Unknown'))
        if subject_str in processed_subjects:
            continue
            
        print(f"\n>>> Processing: {subject_str}")
        raw_result = process_ticket(row)
        
        if raw_result is None:
            print(f"Skipping {subject_str} due to persistent API error.")
            continue

        result_str = str(raw_result)
        
        # Helper to extract structured fields
        def extract(label, default=""):
            if f"{label}:" in result_str:
                try:
                    return result_str.split(f"{label}:")[1].split("\n")[0].strip()
                except IndexError:
                    return default
            return default

        # Map to HackerRank 8-column schema
        new_row = {
            'Issue': row.get('Issue', ''),
            'Subject': row.get('Subject', ''),
            'Company': row.get('Company', ''),
            'Response': result_str.split("Status:")[0].replace("Final Response:", "").strip(),
            'Product Area': extract("Product Area", "General"),
            'Status': extract("Status", "escalated").lower(),
            'Request Type': extract("Request Type", "product_issue").lower(),
            'Justification': extract("Justification", "Resolved via documentation search.")
        }
        
        # Append to CSV
        pd.DataFrame([new_row]).to_csv(OUTPUT_PATH, mode='a', header=not OUTPUT_PATH.exists(), index=False)
        processed_subjects.add(subject_str)
        print(f"Successfully processed: {subject_str}")
        time.sleep(5) # Rate limit protection

def run_single_query():
    """
    Prompts the user for ticket details and processes a single request.
    """
    print("\n--- Single Query Mode ---")
    issue = input("Enter Issue: ").strip()
    subject = input("Enter Subject: ").strip()
    company = input("Enter Company: ").strip()

    if not issue or not subject:
        print("Error: Issue and Subject are required.")
        return

    row = {'Issue': issue, 'Subject': subject, 'Company': company}
    print(f"\n>>> Processing Single Query: {subject}")
    
    raw_result = process_ticket(row)
    if raw_result is None:
        print("Error: Processing failed (API issue).")
        return

    result_str = str(raw_result)

    # Helper to extract structured fields (duplicated for simplicity or could be refactored)
    def extract(label, default=""):
        if f"{label}:" in result_str:
            try:
                return result_str.split(f"{label}:")[1].split("\n")[0].strip()
            except IndexError:
                return default
        return default

    # Extract fields
    response = result_str.split("Status:")[0].replace("Final Response:", "").strip()
    product_area = extract("Product Area", "General")
    status = extract("Status", "escalated").lower()
    request_type = extract("Request Type", "product_issue").lower()
    justification = extract("Justification", "Resolved via documentation search.")

    # Output results line by line
    print("\n--- Results ---")
    print(f"Response: {response}")
    print(f"Product Area: {product_area}")
    print(f"Status: {status}")
    print(f"Request Type: {request_type}")
    print(f"Justification: {justification}")
    print("----------------\n")

if __name__ == "__main__":
    print("Welcome to HackerRank Agent Orchestrator")
    print("1. Batch Processing (support_tickets.csv)")
    print("2. Single Query")
    
    choice = input("\nInput 1 for batch processing and 2 for single query: ").strip()
    
    if choice == '1':
        run_batch_processing()
    elif choice == '2':
        run_single_query()
    else:
        print("Invalid choice. Exiting.")
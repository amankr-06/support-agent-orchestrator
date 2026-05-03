# HackerRank Agent Orchestrator: Multi-Agent Support Intelligence
An automated support resolution system powered by CrewAI and Google Gemini.

This project implements a hierarchical multi-agent system designed to triage, research, and resolve complex support tickets across three major domains: HackerRank, Claude, and Visa. It uses Grounded Retrieval to ensure every response is backed by official company documentation.

---

## Contents

1. [The Problem](#The-Problem)
2. [My Approach](#My-Approach)
3. [Code Structure](#Code-Structure)
4. [Repository Layout](#repository-layout)
5. [Quickstart](#quickstart)

---

## The Problem

Scaling customer support across diverse products (HackerRank, Claude, Visa) is difficult due to **AI hallucinations**, **domain fragmentation**, and **lack of grounding**. Generic chatbots often provide unverified answers that ignore official documentation. 

This project solves these issues by using a multi-agent "Crew" to triage, research, and audit every ticket—ensuring every response is 100% grounded in verified data and meets professional standards.

---

## My Approach

I architected a hierarchical AI support system using the **CrewAI** framework, featuring a **Triage Manager**, a **Search Specialist**, and a **Support Supervisor**.

- **Triage Manager**: Acts as the first point of contact for all incoming support tickets. It analyzes the ticket content, intent, and company context to intelligently route them to the specialized Safety, Search, or Tech domains for resolution.
- **Search Specialist**: The primary knowledge master for the system. It handles FAQ, product inquiries, and technical troubleshooting by searching through the local documentation corpus via the `GroundedSearchTool` to frame detailed, grounded responses.
- **Support Supervisor**: Before any response is finalized, the Supervisor reviews the output. It checks for accuracy, completeness, tone, and adherence to company policies, ensuring the response is 100% grounded in provided documentation without hallucinations.

---

## Code Structure

The core logic of the orchestrator is contained within `code/main.py`, organized into modular sections:

- **Configuration**: Sets up environment variables, project paths, and selects the `gemini-flash-lite` model for cost-efficient, high-speed processing.
- **Custom Tools (`GroundedSearchTool`)**: A specialized tool that performs a secure, local-only search within the `/data` directory. It ensures agents only use verified company documentation.
- **Agent Definitions**: 
  - **Triage Manager**: Orchestrates initial intent classification into Safety, Search, or Tech domains.
  - **Search Specialist**: Handles knowledge retrieval and response drafting using the search tool.
  - **Support Supervisor**: The final auditor for quality, formatting, and grounded truthfulness.
- **Task Orchestration**: Defines a sequential `Crew` process where tasks flow from triage to specialist to supervisor.
- **Execution Engine**:
  - **Batch Mode**: Automated processing for `support_tickets.csv` with built-in resumability and retry logic.
  - **Interactive Mode**: A terminal-based UI for processing single, manual queries.

---

## Repository layout

```
.
├── code/                           # Project dependencies and main script
│   ├── main.py
│   └── requirements.txt
├── .env.example                    # Template for environment variables
├── .gitignore                      # Git exclusion rules
└── README.md                       # Project documentation
```

---

## Quickstart

To get this orchestrator running, follow these steps:

1. **Clone this repository**:
   ```bash
   git clone https://github.com/amankr-06/support-agent-orchestrator.git
   cd support-agent-orchestrator
   ```

2. **Add the Hackathon Data**:
   This repository contains the logic but requires the official dataset to run.
   - Clone the official HackerRank starter repository to a temporary folder:
     ```bash
     git clone git@github.com:interviewstreet/hackerrank-orchestrate-may26.git temp_data
     ```
   - Copy the `data/` and `support_tickets/` folders from `temp_data` into this repository's root.
   - You can now delete the `temp_data` folder.

3. **Configure Environment**:
   Copy `.env.example` to `.env` and add your Gemini API key:
   ```bash
   cp .env.example .env
   # Open .env and add: GEMINI_API_KEY=your_key_here
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r code/requirements.txt
   ```

5. **Run the Orchestrator**:
   ```bash
   python code/main.py
   ```



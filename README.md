# Buildnow Purchase Order Item Categorization Challenge

This repository contains a solution for the Buildnow AI Engineer Internship assessment, focusing on categorizing multilingual purchase order items and analyzing spending patterns.

**The dasboard is hosted currently on the web, you can check it from this link** : 
https://buildnow-assessment.streamlit.app

## Prerequisites

1.  **Python:** Ensure you have Python 3.8 or higher installed.
2.  **Ollama:** This solution uses a locally running Ollama instance to serve the LLM.
    * Install Ollama from [https://ollama.com/](https://ollama.com/).
    * Pull the required Llama model:
        ```bash
        ollama pull llama3:8B
        ```
    * Ensure the Ollama server is running. By default, it runs at `http://localhost:11434`.
3.  **Input Data:** Place the provided `Purchase Order Items.xlsx` file in the root directory of this repository, or update the `INPUT_EXCEL_PATH` variable in the categorization script (`src/categorize_po_items.py` - *adjust filename as needed*) if you place it elsewhere.

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Abdulrazak2004/Buildnow_Assessment
    cd ...
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Ollama URL (Optional):**
    * The script defaults to `http://localhost:11434/v1`.
    * If your Ollama server runs on a different address, set the `OLLAMA_BASE_URL` environment variable:
        ```bash
        # Example for Linux/macOS
        export OLLAMA_BASE_URL='http://your_ollama_ip:11434/v1'
        # Example for Windows (Command Prompt)
        set OLLAMA_BASE_URL=http://your_ollama_ip:11434/v1
        # Example for Windows (PowerShell)
        $env:OLLAMA_BASE_URL='http://your_ollama_ip:11434/v1'
        ```

## Running the Solution

1.  **Run the Categorization Script:**
    * This script loads the Excel data, interacts with the Ollama LLM to categorize items dynamically, refines categories, performs basic analysis, and saves the results to a CSV file (`purchase_orders_categorized_dynamic_v4.csv` by default).
    * *(Assuming your main script is in `src/LLM_Categorizer.py`)*
    ```bash
    python src/categorize_po_items.py
    ```
    * *Note:* This step can take some time depending on the number of items and your hardware (GPUs are recommended for Ollama). Monitor the console output for progress.

2.  **Run the Streamlit Dashboard:**
    * This launches an interactive web application to visualize the categorized data and spending patterns.
    * *(Assuming your Streamlit app is in `src/dashboard.py`)*
    ```bash
    streamlit run src/dashboard.py
    ```
    * Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

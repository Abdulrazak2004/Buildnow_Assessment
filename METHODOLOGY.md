# Methodology: Purchase Order Item Categorization & Spend Analysis MVP

## 1. Introduction & Goal

This document outlines the methodology used to develop a Minimum Viable Product (MVP) for categorizing purchase order (PO) line items and analyzing the resulting spend patterns. The primary goal was to rapidly develop an end-to-end solution that takes raw PO data, leverages a local Large Language Model (LLM) for categorization, and presents insights through an interactive dashboard. This serves as a proof-of-concept demonstrating the potential for automated spend analysis.

## 2. Evolution of the Approach

The development process involved rapid iteration over approximately 48 hours. Initial explorations involved different local LLMs and more complex, multi-stage categorization approaches (including attempts at rule-based merging and secondary LLM-based generalization steps). However, based on observed results and the constraints of using moderately sized local models (like Llama 3 8B), the approach was refined to prioritize robustness and delivering a complete, functional MVP loop. The focus shifted from complex, potentially brittle multi-stage pipelines to a more streamlined, dynamic single-pass categorization followed by a simple clean-up step.

## 3. Current Methodology (v11 Script)

The current approach emphasizes leveraging the LLM for its core strength (understanding text descriptions) while adding simple post-processing for refinement:

1.  **Data Loading and Preparation:**
    * Load data from the specified Excel file (`purchase-order-items.xlsx`) using Pandas.
    * Identify and standardize essential columns (Item ID, Item Description, Spending Amount).
    * Perform basic cleaning: handle missing descriptions, ensure unique Item IDs (appending indices if duplicates are found), and standardize text formatting.

2.  **Initial Category Suggestion (LLM-driven):**
    * To provide a relevant starting point, a random sample of unique item descriptions (up to `INITIAL_CATEGORY_SAMPLE_SIZE`, e.g., 1000-2500) is sent to the LLM (`Llama3:8B`).
    * The LLM is prompted to suggest a list of reasonably specific, distinct procurement categories based on the sample, returning them in JSON format.
    * Essential categories ("Other", "Unknown") are added if not suggested by the LLM. This list forms the initial seed for categorization.

3.  **Dynamic Item Categorization (LLM-driven):**
    * Each item description is processed individually (in parallel using `concurrent.futures.ThreadPoolExecutor` for efficiency).
    * For each item, the LLM is prompted to assign the most appropriate category. It is shown the *current list* of known categories (including those dynamically added by other parallel threads) and instructed to:
        * Prefer existing categories.
        * Suggest a *new*, concise category only if the item represents a truly distinct group not already covered.
        * Use "Other" or "Unknown" appropriately.
        * Ignore minor variations (size, brand, etc.).
    * A shared set (`live_categories_set`) protected by a thread lock (`category_lock`) tracks all discovered categories (suggested + dynamically added) in real-time, ensuring consistency across parallel threads.

4.  **Low-Frequency Category Merging:**
    * After all items are categorized, a simple post-processing step is applied.
    * The script calculates the frequency of each assigned category.
    * Categories with a count less than or equal to a defined threshold (`LOW_COUNT_THRESHOLD`, e.g., 10) are automatically re-assigned to the "Other" category.
    * This step helps consolidate very sparse categories, cleaning up potential noise or overly specific suggestions from the LLM without requiring complex rules or another LLM pass.

5.  **Analysis and Output:**
    * The final DataFrame, containing the original data plus the cleaned `Category` column, is analyzed.
    * Value counts and optional spending totals per category are calculated and printed.
    * The results are saved to a CSV file (`purchase_orders_categorized.csv`) with UTF-8-SIG encoding for compatibility.

## 4. Technology Stack

* **Backend/Processing:** Python 3.x
* **Data Handling:** Pandas
* **LLM Interaction:** Ollama, `openai` Python client
* **LLM Model:** Llama 3 8B (configurable via `Llama3:8B`)
* **Dashboard:** Streamlit
* **Visualization:** Plotly Express

## 5. Role of LLMs

LLMs played a central role in both the *process* and the *product*:

* **Core Categorization:** The selected `Llama3:8B` (e.g., Llama 3 8B) running via Ollama performs the primary task of understanding item descriptions and assigning categories, both initially and dynamically.
* **Development Assistance:** Generative AI tools (like ChatGpt 3.5) were utilized during development for:
    * **Boilerplate Code:** Generating repetitive code structures.
    * **Prompt Engineering:** Assisting in drafting and refining prompts sent to the categorization LLM.
    * **Code Explanation & Commenting:** Adding clarifying comments and explanations to improve code readability and maintainability.
    * **Refactoring:** Suggesting improvements for code structure and clarity.


## 6. Interactive Dashboard

A Streamlit application (`dashboard.py`) was developed to visualize the categorized data:

* **Overview:** Provides an interactive interface to explore the spend data based on the LLM-generated categories.
* **Key Features:**
    * **KPIs:** Displays summary statistics like Total Spend, Total Items, Total Quantity, Unique Categories, and Average Spend per Item.
        ![Dashboard KPI Row](https://github.com/Abdulrazzak-Ghazal/Buildnow_Assessment/raw/master/image_folder/1.png)
    * **Filtering:** Allows users to filter the data by Project ID, Purchase Order ID, and Category via sidebar controls.
        ![Filtering Functionality](https://github.com/Abdulrazzak-Ghazal/Buildnow_Assessment/raw/master/image_folder/2.png)
    * **Distribution Charts:** Shows total spend and total quantity per category using Plotly bar charts.
        ![Charts, Visualization](https://github.com/Abdulrazzak-Ghazal/Buildnow_Assessment/raw/master/image_folder/3.png)
    * **Detailed View:**
        * If a specific PO is selected, it displays the items within that PO.
        * Otherwise, it allows users to select a category and view the top items within it by spend and quantity.
        ![Detailed Tables](https://github.com/Abdulrazzak-Ghazal/Buildnow_Assessment/raw/master/image_folder/4.png)
    * **Project Analysis:** If specific projects are filtered, shows category spending and top items within those projects.
    * **Data Explorer:** Provides access to the raw (filtered) data in a table format.


## 7. Limitations and Future Work

This MVP demonstrates feasibility but has limitations:

* **LLM Dependency:** The quality of categorization is highly dependent on the chosen LLM (`Llama3:8B`) and the quality of the prompts.
* **Hardware Constraints:** Running larger, potentially more accurate LLMs was limited by available GPU resources. Performance and accuracy could be significantly enhanced with higher-end GPUs allowing for larger models or fine-tuning.
* **Simplistic Merging:** The low-count category merge is basic. More sophisticated clustering or rule-based merging could be explored.
* **Scalability:** While parallel processing helps, categorizing extremely large datasets might require further optimization or distributed processing.
* **Contextual Understanding:** The LLM relies primarily on item descriptions. Incorporating other data fields (supplier, unit price, etc.) could improve accuracy.

Future work could involve:
* Experimenting with larger or fine-tuned LLMs.
* Developing more sophisticated category refinement techniques.
* Integrating supplier analysis.
* Building more advanced dashboard features (time-series analysis, anomaly detection).

## 8. Conclusion

This project successfully delivered an MVP for LLM-powered PO categorization and spend analysis within a short timeframe. By focusing on a streamlined dynamic categorization approach and leveraging LLMs for both the core task and development assistance, a valuable proof-of-concept was created. The resulting dashboard provides immediate insights into spending patterns.

The iterative process highlighted the importance of adapting the methodology based on tool capabilities and constraints. This exercise demonstrates a strong capability and interest in applying AI and data analysis techniques to solve real-world business problems, like those faced by Buildnow, and delivering tangible results quickly.

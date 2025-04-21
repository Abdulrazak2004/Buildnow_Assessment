# -*- coding: utf-8 -*-
"""
Script for categorizing purchase order items using an LLM (Ollama).

This script loads purchase order data, cleans it, uses an LLM to suggest
initial categories based on a sample, then categorizes each item dynamically
using the LLM in parallel. New categories suggested by the LLM are added
on the fly. Finally, it merges low-frequency categories and saves the
results with optional spending analysis.
"""

import pandas as pd
from openai import OpenAI
import time
import random
import re
import concurrent.futures
import threading
# I'm importing json here cause the LLM is prompted to return JSON for initial suggestions,
# and I need to parse that response. This ensures easy extraction of the suggested categories.
import json
# I'm importing os here cause I need to access environment variables (for the Ollama URL)
# and check if the input file exists. This ensures the script is configurable and robust against missing files.
import os
# I'm keeping defaultdict commented out as it's not currently used, but might be useful for future aggregation tasks.
# from collections import Counter, defaultdict # Keep defaultdict if needed elsewhere

# --- Configuration Constants ---
# These constants centralize settings, making the script easier to configure.

# --- LLM & API Settings ---
# I'm defining the environment variable name for the Ollama URL here cause it allows users
# to configure the URL without modifying the script directly. This ensures flexibility.
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
# I'm setting a default Ollama URL here cause it provides a sensible fallback if the
# environment variable isn't set. This ensures the script works out-of-the-box for common setups.
DEFAULT_OLLAMA_URL = 'http://localhost:11434/v1'
# I'm specifying the LLM model name here cause this model will be used for both suggesting
# initial categories and performing the item-by-item categorization. This ensures consistency.
LLM_MODEL_NAME = "llama3:8B"
# I'm defining the maximum number of concurrent worker threads here cause LLM API calls are I/O bound,
# and parallel execution significantly speeds up processing. This ensures faster categorization,
# but should be adjusted based on the Ollama server's capacity and the machine running the script.
MAX_WORKERS = 64
# I'm setting a delay between retries for failed API calls here cause hitting the API too quickly
# after a failure might exacerbate the issue (e.g., rate limiting). This ensures a gentler retry mechanism.
RETRY_DELAY = 2
# I'm defining the maximum number of retries for a failed API call here cause network issues or
# temporary server problems might occur. This ensures resilience against transient errors.
MAX_RETRIES = 3

# --- File Paths ---
# *** IMPORTANT: Make sure this path points to your actual Excel file ***
# I'm defining the input file path here cause the script needs to know where to load the data from.
# This ensures the user can easily specify their input file.
INPUT_EXCEL_PATH = "purchase-order-items.xlsx"
# I'm defining the output file path here cause the script needs a destination for the categorized data.
# This ensures the results are saved predictably.
OUTPUT_CSV_PATH = "purchase_orders_categorized.csv"

# --- Categorization Logic Settings ---
# I'm defining the sample size for initial category suggestion here cause analyzing all unique items
# might be slow or unnecessary for getting a good starting list. This ensures a balance between
# representativeness and speed for the initial LLM call.
INITIAL_CATEGORY_SAMPLE_SIZE = 2500
# I'm defining a threshold for merging low-count categories here cause very infrequent categories
# might represent noise or minor variations better grouped under 'Other'. This ensures a cleaner final category list.
LOW_COUNT_THRESHOLD = 1
# I removed the fixed target list here cause this version dynamically discovers categories,
# starting with LLM suggestions and adding more as needed during categorization. This ensures flexibility
# and adaptability to the specific data encountered.

# --- End Configuration Constants ---

# --- Shared Resources for Concurrency ---
# I'm creating a threading Lock here cause the 'live_categories_set' will be accessed and modified
# by multiple threads concurrently during categorization. This ensures thread safety, preventing race
# conditions where multiple threads might try to read/write the set simultaneously, leading to errors or inconsistencies.
category_lock = threading.Lock()
# I'm initializing an empty set to store categories dynamically here cause as the LLM categorizes items,
# it might suggest new categories not in the initial list. This set will store all discovered categories
# (in lower case for case-insensitive checking) in a thread-safe manner (using the lock). This ensures
# that subsequent LLM calls are aware of newly added categories.
live_categories_set = set()
# --- End Shared Resources ---

# --- LLM Client Setup ---
print("--- Initializing LLM Client ---")
try:
    # I'm getting the Ollama base URL from an environment variable first, then using the default,
    # cause this provides flexibility for users running Ollama on different ports or machines. This ensures easy configuration.
    ollama_base_url = os.getenv(OLLAMA_BASE_URL_ENV, DEFAULT_OLLAMA_URL)
    # I'm initializing the OpenAI client, pointing it to the local Ollama URL, cause this is how
    # we interact with the Ollama server using the OpenAI library format. The API key is nominal for local Ollama.
    # This ensures the script can communicate with the specified LLM server.
    client = OpenAI(base_url=ollama_base_url, api_key='ollama')
    # I'm making a test call to list models here cause it verifies that the connection to the
    # Ollama server is successful before proceeding. This ensures early failure detection if the server is down or inaccessible.
    client.models.list()
    print(f"Successfully connected to Ollama server at {ollama_base_url}.")
    print(f"Using LLM Model for suggestion & categorization: {LLM_MODEL_NAME}")

except Exception as e:
    # I'm catching exceptions during client setup here cause connection issues are common.
    # This ensures the script provides informative error messages and exits gracefully if it can't connect.
    print(f"Error connecting to Ollama server at {ollama_base_url}: {e}")
    print("Please ensure Ollama is running and accessible.")
    print(f"You can set the {OLLAMA_BASE_URL_ENV} environment variable if it's not running on the default location.")
    exit()
# --- End LLM Client Setup ---

# --- Function Definitions ---

def clean_category_name(category_text):
    """
    Cleans and standardizes the category name received from the LLM.

    I created this function cause LLM responses can be inconsistent, containing
    extra phrases, quotes, or weird formatting. This ensures that category names
    are standardized before being used or saved.

    Args:
        category_text (str): The raw category text from the LLM.

    Returns:
        str: The cleaned and standardized category name (e.g., "Unknown", "Other", or Title Case name).
    """
    # I'm checking for None or NaN here cause pandas might introduce NaNs.
    # This ensures the function handles missing inputs gracefully, returning "Unknown".
    if not category_text or pd.isna(category_text):
        return "Unknown"

    # I'm converting to string and stripping whitespace here cause the input might not be a string,
    # and leading/trailing spaces are common. This ensures consistent handling.
    cleaned = str(category_text).strip()
    # I'm removing potential quotes here cause LLMs sometimes wrap responses in quotes.
    # This ensures quotes aren't part of the category name.
    cleaned = cleaned.replace('"', '').replace("'", "")
    # I'm using regex to remove common introductory phrases (case-insensitive) here cause
    # LLMs might add text like "Based on the item description, I would categorize it as:".
    # This ensures only the category name remains.
    cleaned = re.sub(r"(?i)^based on the item description,?.*i would categorize it as:?", "", cleaned).strip()
    cleaned = re.sub(r"(?i)^the category is:?", "", cleaned).strip()
    cleaned = re.sub(r"(?i)^category:?", "", cleaned).strip()
    # I'm removing leading/trailing non-alphanumeric characters (allowing spaces, slashes, hyphens)
    # here cause sometimes stray punctuation appears. This ensures cleaner category names.
    cleaned = re.sub(r"^[^\w\s\/-]+|[^\w\s\/-]+$", "", cleaned).strip()
    # I'm converting the name to Title Case here cause it provides a consistent capitalization format.
    # This ensures uniformity in the final category list.
    cleaned = ' '.join(word.capitalize() for word in cleaned.split()).strip()

    # I'm checking for empty strings after cleaning here cause the cleaning steps might remove everything.
    # This ensures we return "Unknown" instead of an empty category.
    if not cleaned: return "Unknown"

    # I'm performing case-insensitive checks for core categories here cause the LLM might return
    # "unknown" or "other" in different cases. This ensures these core categories are always standardized.
    cleaned_lower = cleaned.lower()
    if cleaned_lower == 'unknown': return 'Unknown'
    if cleaned_lower == 'other': return 'Other'
    if cleaned_lower == 'error': return 'Error' # Added for consistency

    # I'm removing potential numbered list prefixes (e.g., "1. ") here cause the LLM might format
    # suggestions as lists. This ensures the number is not part of the category name.
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)

    # I'm truncating excessively long category names here cause overly specific or verbose names
    # are usually undesirable. This ensures category names remain reasonably concise.
    if len(cleaned) > 60:
        print(f"Warning: Truncating overly long cleaned category name: '{cleaned}'")
        cleaned = cleaned[:60].strip()

    return cleaned

def suggest_initial_categories(sample_descriptions, client, model_name):
    """
    Asks the LLM to suggest an initial list of procurement categories based on a sample of item descriptions.

    I created this function cause starting with a relevant set of categories improves the subsequent
    item-by-item categorization process. This ensures the dynamic categorization has a sensible baseline.

    Args:
        sample_descriptions (list): A list of unique item descriptions to sample from.
        client (OpenAI): The initialized OpenAI client configured for Ollama.
        model_name (str): The name of the LLM model to use.

    Returns:
        list or None: A list of suggested category names (strings), or None if an error occurs.
    """
    print(f"\n--- Requesting Initial Category Suggestions from LLM based on {len(sample_descriptions)} samples ---")
    # I'm checking if the sample list is empty here cause the function cannot proceed without data.
    # This ensures a safe exit if no samples are provided.
    if not sample_descriptions:
        print("Warning: No sample descriptions provided. Cannot suggest categories.")
        return None

    # I'm creating a string from a random sample of descriptions here, limited to 50 items,
    # cause the LLM prompt shouldn't be excessively long or expensive.
    # This ensures the LLM gets representative examples without exceeding token limits or context windows.
    sample_str = "\n- ".join(random.sample(sample_descriptions, min(len(sample_descriptions), 50)))

    # I'm constructing the prompt here, providing context (SMEs, Saudi Arabia, data mix) and clear instructions
    # (suggest categories, focus on core types, provide JSON output), cause well-defined prompts lead to better LLM responses.
    # This ensures the LLM understands the task and the desired output format.
    prompt = f"""
    Based on the following sample of purchase order item descriptions from construction/manufacturing/business SMEs in Saudi Arabia, suggest a list of relevant and distinct procurement categories.
    The data includes items like materials, tools, equipment, safety gear, office supplies, IT hardware/software, furniture, vehicle parts, services, consumables, building materials, etc. Descriptions are mixed English/Arabic/Transliterated.
    Aim for categories that are reasonably specific at this stage but avoid excessive granularity (e.g., prefer "Steel Pipes" over "3-inch Galvanized Steel Pipe"). The goal is to capture the primary types of items purchased. Focus on the core item type, ignoring minor variations in size, brand, or specific grade unless it defines a major sub-type.

    Sample Item Descriptions:
    - {sample_str}

    Provide your suggestions as a SINGLE JSON object with one key "suggested_categories", which holds a list of strings (the category names in English). Example: {{"suggested_categories": ["Steel Products", "Safety Shoes", "Office Furniture", "Consulting Services", "Electrical Wiring", "Concrete Blocks"]}}
    Ensure category names are concise (2-4 words). Include general categories like "Other" and "Unknown".

    Return ONLY the JSON object. Ensure the JSON is valid.
    JSON Suggestions:
    """
    response_content = "" # Initialize to handle potential errors
    try:
        # I'm calling the LLM's chat completion endpoint here cause it's suitable for instruction-following tasks.
        # I specified the model, messages (system prompt for role, user prompt for task), temperature (low for consistency),
        # max_tokens, and importantly, response_format={"type": "json_object"}.
        # This ensures the LLM attempts to return valid JSON, simplifying parsing.
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert procurement analyst creating an initial list of categories based on sample data for SMEs. Focus on clear, common procurement groups, ignoring minor variations. Output ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, # Slightly creative but mostly consistent
            max_tokens=1000, # Allow enough space for a decent list
            response_format={"type": "json_object"} # Request JSON output
        )
        response_content = response.choices[0].message.content.strip()

        # I'm parsing the LLM response as JSON here cause the prompt requested JSON format.
        # This ensures easy extraction of the suggested list.
        suggestions_data = json.loads(response_content)

        # I'm checking if the expected key and list type are present in the parsed JSON
        # cause the LLM might not perfectly follow instructions. This ensures robust parsing.
        if "suggested_categories" in suggestions_data and isinstance(suggestions_data["suggested_categories"], list):
            # I'm cleaning each suggested category using the clean_category_name function and filtering empty ones
            # cause the LLM might still include minor inconsistencies or empty strings. This ensures a clean list.
            suggested_list = [clean_category_name(cat) for cat in suggestions_data["suggested_categories"] if cat]
            # I'm explicitly adding "Unknown" and "Other" if they aren't already present cause these are essential
            # fallback categories. This ensures they are always available options.
            if "Unknown" not in suggested_list: suggested_list.append("Unknown")
            if "Other" not in suggested_list: suggested_list.append("Other")
            # I'm removing potential duplicates and sorting the list here cause the cleaning or LLM might introduce them.
            # This ensures a unique, sorted list for better readability and consistency.
            suggested_list = sorted(list(set(filter(None, suggested_list)))) # Filter out potential empty strings after cleaning
            print(f"LLM Suggested Initial Categories: {suggested_list}")
            return suggested_list
        else:
            # I'm handling the case where the JSON structure is incorrect here cause LLM guarantees aren't perfect.
            # This ensures the script reports the issue instead of crashing.
            print("Warning: LLM response did not contain a valid 'suggested_categories' list in JSON.")
            print(f"LLM Raw Response: {response_content}")
            return None
    except json.JSONDecodeError:
        # I'm handling JSON parsing errors specifically here cause this is a common failure point when expecting JSON from LLMs.
        # This ensures the raw response is printed for debugging.
        print("Error: LLM did not return valid JSON for initial categories. Raw output:")
        print(response_content)
        return None
    except Exception as e:
        # I'm catching any other exceptions during the API call or processing here cause various issues can occur.
        # This ensures general errors are caught and reported.
        print(f"Error during initial category suggestion: {e}")
        return None

def categorize_item_dynamic(item_id, description, current_categories_set_lower, lock, client, model_name):
    """
    Categorizes a single item description using the LLM, allowing dynamic addition of new categories.

    I created this function to be run in parallel for each item cause it's the core, time-consuming
    part of the script. It leverages the LLM to classify items based on existing categories or suggest
    new ones if appropriate, while carefully managing shared state (the category list). This ensures
    efficient processing and discovery of relevant categories within the dataset.

    Args:
        item_id (str): The unique identifier for the item.
        description (str): The item description text.
        current_categories_set_lower (set): The shared set holding lowercased discovered categories.
        lock (threading.Lock): The lock to protect access to current_categories_set_lower.
        client (OpenAI): The initialized OpenAI client.
        model_name (str): The name of the LLM model to use.

    Returns:
        tuple: A tuple containing (item_id, final_category_name).
               Returns (item_id, "Unknown") for empty descriptions or (item_id, "Error") on failure.
    """
    # I'm checking for empty or NaN descriptions at the start here cause such items cannot be categorized.
    # This ensures they are immediately assigned "Unknown" without calling the LLM.
    if not description or pd.isna(description) or str(description).strip() == "":
        return item_id, "Unknown"

    # I'm acquiring the lock *before* reading the shared category set here cause another thread might be
    # modifying it concurrently. This ensures the list used in the prompt reflects a consistent snapshot
    # of the categories discovered so far.
    with lock:
        # I'm converting the lowercased set back to Title Case for the prompt here cause it's more readable for the LLM.
        # I'm excluding core/utility categories from the main list in the prompt cause they are handled by specific instructions.
        # This ensures the prompt focuses on the domain-specific categories.
        category_list_for_prompt = sorted([cat.title() for cat in current_categories_set_lower if cat not in ["unknown", "other", "error"]])
        # I'm always adding "Other" to the list presented in the prompt cause it's a valid choice the LLM should consider.
        # This ensures the LLM is explicitly aware of the 'Other' option.
        category_list_for_prompt.append("Other")
        # I'm ensuring the list is unique (set) and sorted again after adding 'Other' cause this maintains consistency.
        category_list_for_prompt = sorted(list(set(category_list_for_prompt)))

    # I'm formatting the category list into a string for the prompt here cause the LLM needs to see the available options.
    # This ensures the LLM knows which categories it can choose from.
    category_list_str = ', '.join(f'"{cat}"' for cat in category_list_for_prompt) if category_list_for_prompt != ["Other"] else "No specific categories yet, suggest one or use Other/Unknown."

    # I'm constructing a detailed prompt here with specific instructions cause guiding the LLM carefully is crucial
    # for getting accurate and consistent categorization, especially regarding when to suggest new categories vs. using existing ones.
    # This ensures the LLM focuses on the core item type, avoids creating categories for minor variations, and uses the provided list primarily.
    prompt = f"""
    Analyze the purchase order item description below (mixed English/Arabic/Transliterated). Context: Construction/Manufacturing/Business SMEs in Saudi Arabia.
    Your task is to assign the MOST appropriate category for this specific item, focusing on its **core type or function**.

    **IMPORTANT INSTRUCTIONS:**
    1.  **Ignore Minor Variations:** Do NOT create new categories based on differences in size, thickness, specific material grade, color, brand, or supplier unless it defines a fundamentally different *type* of item.
    2.  **Focus on Class:** Group similar items (e.g., 'Steel Pipe 12mm' and 'Steel Pipe 6 inch' should likely be 'Steel Pipes' or similar).
    3.  **Use Existing Categories First:** If the item clearly fits one of these existing categories (even broadly), choose EXACTLY ONE from this list:
        [ {category_list_str} ]
    4.  **Suggest New Category (Only if Necessary):** If the item represents a truly distinct group NOT well represented by the list (and isn't just 'Other' or a minor variation), suggest a concise NEW category name in English (2-4 words, e.g., "Scaffolding Components", "Rental Equipment", "Welding Consumables").
    5.  **Use "Unknown":** Only if the description is genuinely unintelligible. Minimize using "Unknown".
    6.  **Use "Other":** If the item doesn't fit existing categories and doesn't warrant a new distinct category.

    **Output Format:** Output ONLY the single chosen or suggested category name. Do NOT add explanations or any other text.

    Item Description: "{description}"

    Category:
    """
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            # I'm calling the LLM here with a very low temperature (0.05) cause categorization should be deterministic
            # and consistent, rather than creative. I'm also setting a low max_tokens cause only the category name is expected.
            # This ensures reliable and concise output.
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an assistant categorizing procurement items. Focus on core item type, ignore minor variations. Select from list, suggest new category ONLY if truly distinct, or use 'Unknown'/'Other'. Output ONLY the category name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05, # Low temperature for more deterministic classification
                max_tokens=50 # Short response expected (just the category)
            )
            llm_output_raw = response.choices[0].message.content.strip()
            # I'm cleaning the LLM's response using the dedicated function here cause the raw output might still need standardization.
            # This ensures the category name is in the correct format.
            category_suggestion = clean_category_name(llm_output_raw)

            # I'm determining the final category based on the cleaned suggestion here, mapping specific cleaned results
            # to the standard "Unknown", "Other", or "Error". This ensures consistent handling of these special cases.
            if not category_suggestion or category_suggestion == "Unknown":
                final_category = "Unknown"
            elif category_suggestion == "Other":
                final_category = "Other"
            elif category_suggestion == "Error":
                final_category = "Error" # Should ideally not happen via LLM but handled if cleaning produces it
            else:
                # If it's not Unknown/Other/Error, it's either an existing category or a potential new one.
                final_category = category_suggestion

            # --- Thread-safe check and update of the live category set ---
            # I'm acquiring the lock again here before checking and potentially modifying the shared 'live_categories_set'
            # cause this is the critical section where shared state is updated. This ensures that the check and subsequent
            # addition (if needed) happen atomically, preventing race conditions.
            with lock:
                # I'm getting a direct reference to the live set *inside the lock* for the most up-to-date check.
                # Note: Directly using live_categories_set is safe here *because* we hold the lock.
                final_category_lower = final_category.lower()
                # I'm adding the category to the shared set only if it's a *new*, *valid* category
                # (i.e., not Unknown/Other/Error and not already present) cause we only want to track newly discovered categories.
                # This ensures the 'live_categories_set' grows dynamically and safely.
                if final_category_lower not in ["unknown", "other", "error"] and final_category_lower not in live_categories_set:
                    print(f"Dynamically adding new category: '{final_category}' (from item ID {item_id})")
                    live_categories_set.add(final_category_lower) # Add the new category (lower case)
                # I'm also ensuring the core categories ('unknown', 'other', 'error') get added to the set
                # the first time they are assigned by any thread cause we need them to be in the final list if used.
                # This ensures completeness of the tracked categories.
                elif final_category_lower in ["unknown", "other", "error"] and final_category_lower not in live_categories_set:
                     live_categories_set.add(final_category_lower)

            # I'm returning the category with its original capitalization (as cleaned) here cause that's the desired format for the DataFrame.
            # This ensures the final output uses the Title Case (or specific cases like Unknown/Other).
            return item_id, final_category

        except Exception as e:
            # I'm handling exceptions during the API call or response processing here cause errors can occur.
            # I increment the retry counter. This ensures transient issues are retried.
            retries += 1
            print(f"Warning: API call failed for item ID {item_id} (Attempt {retries}/{MAX_RETRIES}). Error: {e}")
            # I'm checking if max retries have been exceeded here cause we need to stop eventually.
            # This ensures the process doesn't hang indefinitely on a persistent error.
            if retries > MAX_RETRIES:
                print(f"Error: Max retries exceeded for item ID {item_id}. Assigning 'Error'.")
                # I'm acquiring the lock before potentially adding 'error' to the set cause this assignment
                # also needs to update the shared state if 'error' wasn't seen before. This ensures thread safety.
                with lock:
                   if "error" not in live_categories_set: live_categories_set.add("error")
                return item_id, "Error" # Return 'Error' category after max retries
            # I'm introducing a delay before the next retry, increasing the delay with each attempt (exponential backoff),
            # cause this reduces load on the server and increases the chance of success if the issue is temporary.
            # This ensures a robust retry strategy.
            time.sleep(RETRY_DELAY * (retries + 1)) # Use exponential backoff

    # I'm adding a fallback return statement here just in case the loop finishes unexpectedly (shouldn't happen with the logic).
    # This ensures the function always returns a tuple, assigning 'Error'.
    print(f"Error: Failed to categorize item ID {item_id} after multiple retries. Assigning 'Error'.")
    with lock: # Ensure 'Error' is in the set if assigned via this fallback
        if "error" not in live_categories_set: live_categories_set.add("error")
    return item_id, "Error"

# --- Main Script Execution ---

# === Step 1: Load Data ===
print(f"\n--- Loading Data ---")
# I'm checking if the input Excel file exists before trying to load it here cause attempting to load
# a non-existent file raises an error. This ensures a user-friendly error message if the path is wrong.
if not os.path.exists(INPUT_EXCEL_PATH):
    print(f"Error: Input Excel file not found at {INPUT_EXCEL_PATH}")
    exit()
try:
    # I'm loading the data from the specified Excel file into a pandas DataFrame here cause pandas
    # is efficient for handling tabular data. This ensures the data is ready for processing.
    df = pd.read_excel(INPUT_EXCEL_PATH)
    print(f"Successfully loaded {len(df)} rows from Excel file: {INPUT_EXCEL_PATH}")
except Exception as e:
    # I'm catching potential errors during file loading (e.g., corrupted file, wrong format, missing dependency)
    # here cause file I/O can fail. This ensures graceful failure with helpful advice.
    print(f"Error loading Excel file: {e}")
    print("Make sure the file path is correct, the file is a valid Excel file,")
    print("and you have the required library installed (`pip install openpyxl` or `pip install xlrd` depending on the file format).")
    exit()

# === Step 2: Initial Data Cleaning & Preparation ===
print("\n--- Cleaning and Preparing Data ---")
initial_rows = len(df)

# --- Standardize Item Description Column ---
# I'm searching for common names for the item description column here cause different files might use different headers.
# This ensures the script can find the relevant data column automatically in many cases.
item_name_col = None
possible_desc_cols = ['Item Name', 'Description', 'Item Description', 'PO Item Description', 'Material Description', 'Service Description']
for col in possible_desc_cols:
    if col in df.columns:
        item_name_col = col
        # I'm renaming the found column to a standard 'Item Name' here cause it simplifies the rest of the script
        # which can then rely on this consistent column name. This ensures easier code maintenance.
        if col != 'Item Name':
            df.rename(columns={col: 'Item Name'}, inplace=True)
            print(f"Using column '{col}' as 'Item Name'.")
        else:
             print(f"Using column 'Item Name' for descriptions.")
        break # Stop searching once found

# I'm checking if an item description column was actually found here cause the script cannot proceed without it.
# This ensures the script exits gracefully if the essential data is missing.
if not item_name_col:
    print(f"Error: Could not find a suitable item description column among {possible_desc_cols}. Please check your Excel file headers.")
    exit()

# I'm dropping rows where the 'Item Name' is missing (NaN) here cause these rows cannot be categorized.
# This ensures we don't process invalid entries.
df.dropna(subset=['Item Name'], inplace=True)
# I'm converting the 'Item Name' column to string and stripping whitespace here cause descriptions might be numeric
# or have extra spaces. This ensures consistent text formatting for the LLM.
df['Item Name'] = df['Item Name'].astype(str).str.strip()
# I'm filtering out rows where the 'Item Name' became an empty string after stripping here cause these are also unusable.
# This ensures only rows with actual descriptions remain.
df = df[df['Item Name'] != '']
cleaned_rows = len(df)
rows_dropped = initial_rows - cleaned_rows
if rows_dropped > 0:
    print(f"Dropped {rows_dropped} rows due to missing/empty 'Item Name'.")
print(f"Remaining rows for processing: {cleaned_rows}")

# --- Ensure Unique Item ID Column ---
# I'm searching for common names for an item ID column here cause a unique identifier is needed
# to map the categorization results back correctly, especially with parallel processing.
# This ensures each original row can be identified.
id_col = None
possible_id_cols = ['Item ID', 'PO Item ID', 'Line Item ID', 'ID', 'Unique ID']
for col in possible_id_cols:
    if col in df.columns:
        id_col = col
        # I'm renaming the found ID column to a standard 'Item ID' here cause it simplifies referencing it later.
        # This ensures consistency.
        if col != 'Item ID':
            df.rename(columns={col: 'Item ID'}, inplace=True)
            print(f"Using column '{col}' as 'Item ID'.")
        else:
             print(f"Using column 'Item ID' as the unique identifier.")
        break

# I'm creating a sequential 'Item ID' if none was found here cause a unique ID is essential for the script's logic.
# This ensures every row has an identifier, even if the original data lacked one.
if not id_col:
    print("Warning: No standard 'Item ID' column found. Creating sequential IDs starting from 0.")
    df.insert(0, 'Item ID', range(len(df)))
    id_col = 'Item ID' # Set the name of the newly created column

# I'm converting the 'Item ID' column to string here cause mixing numeric and string IDs can cause issues,
# especially if we need to modify them (like appending indices for duplicates). This ensures consistent ID type.
df['Item ID'] = df['Item ID'].astype(str)

# I'm checking for and handling duplicate 'Item ID's here cause duplicates would prevent correct mapping of results.
# I append the DataFrame index to duplicates to make them unique. This ensures every ID is truly unique before processing.
if df['Item ID'].duplicated().any():
    print("Warning: Duplicate values found in 'Item ID' column. Appending index to duplicates to ensure uniqueness.")
    # Reset index to make the index available as a column temporarily
    df.reset_index(inplace=True)
    # Apply a function to append '_index' only to rows where the original 'Item ID' was duplicated
    df['Item ID'] = df.apply(
        lambda row: f"{row['Item ID']}_{row['index']}" if df['Item ID'].duplicated(keep=False)[row.name] else str(row['Item ID']),
        axis=1
    )
    # Drop the temporary index column
    df.drop(columns=['index'], inplace=True)
    print("Duplicate 'Item ID's have been made unique.")


# === Step 3: Determine Initial Categories via LLM Suggestion ===
print("\n--- Determining Initial Categories (using LLM suggestion) ---")
# I'm getting unique item descriptions here cause we only need to sample from unique values for suggesting categories.
# This ensures the sample is diverse and avoids redundancy.
unique_items = df['Item Name'].unique()
# I'm determining the sample size, capped by the configuration constant or the actual number of unique items,
# cause we need a reasonable number of examples for the LLM. This ensures we don't try to sample more items than exist.
sample_size = min(len(unique_items), INITIAL_CATEGORY_SAMPLE_SIZE)
# I'm taking a random sample of unique item descriptions here cause this provides the LLM with varied examples.
# This ensures the sample is representative (within the size limit).
item_sample = random.sample(list(unique_items), sample_size) if sample_size > 0 else []

# I'm calling the function to get initial category suggestions from the LLM here cause this provides
# a starting point for the dynamic categorization process. This ensures a data-driven initial list.
suggested_categories_list = suggest_initial_categories(item_sample, client, LLM_MODEL_NAME)

# I'm defining a fallback list of categories here cause the LLM suggestion might fail (network error, bad response).
# This ensures the script can continue with a reasonable default list if the LLM suggestion step fails.
if not suggested_categories_list:
    print("LLM suggestion failed or returned no categories. Using a default fallback list.")
    # This fallback list contains common procurement categories for the target domain.
    suggested_categories_list = [
        "Raw Materials", "Steel Products", "Piping & Fittings", "Building Materials",
        "Fasteners & Hardware", "Hand Tools", "Power Tools", "Heavy Equipment",
        "Safety Equipment", "Electrical Components", "Instrumentation & Controls",
        "Plumbing Supplies", "HVAC Supplies", "Chemicals & Adhesives",
        "Office Supplies", "IT Equipment & Software", "Logistics & Transport",
        "Maintenance & Repair Services", "Consulting & Professional Services",
        "Consumables", "Vehicle Parts & Maintenance", "Welding Supplies",
        "Testing Equipment", "Surveying Equipment", "Furniture", "Aggregates & Concrete",
        "Rentals", "Lubricants & Greases", "Catering & Food Supplies",
        "Other", "Unknown"
    ]
    # Ensure core categories are present in the fallback list too
    if "Unknown" not in suggested_categories_list: suggested_categories_list.append("Unknown")
    if "Other" not in suggested_categories_list: suggested_categories_list.append("Other")
    suggested_categories_list = sorted(list(set(suggested_categories_list)))


# I'm clearing and initializing the shared 'live_categories_set' with the lowercased starting categories
# (either from LLM or fallback) here cause this set will track all known categories during the parallel processing phase.
# Using lower case ensures case-insensitive checking. This ensures the categorization threads start with the initial list.
live_categories_set.clear() # Ensure it's empty before starting
live_categories_set.update([cat.lower() for cat in suggested_categories_list])
print(f"\nStarting dynamic categorization with initial categories (lower case): {sorted(list(live_categories_set))}")
# Print Title Case for readability
print(f"Starting dynamic categorization with initial categories (Title Case): {sorted([cat.title() for cat in live_categories_set])}")


# === Step 4: Prepare Data for Parallel Processing ===
# I'm creating a list of tuples (Item ID, Item Name) here cause this format is convenient
# for submitting tasks to the ThreadPoolExecutor. This ensures the necessary data is ready for parallel processing.
items_to_process = list(df[['Item ID', 'Item Name']].itertuples(index=False, name=None))


# === Step 5: Categorize All Items Dynamically (Single Pass - Parallel Processing) ===
print(f"\n--- Starting Dynamic Categorization ({len(items_to_process)} items, max {MAX_WORKERS} workers) ---")
# I'm initializing a dictionary to store the results (mapping Item ID to Category) here cause
# threads will complete out of order, and we need a way to collect the results associated with the correct item.
# This ensures results are correctly mapped back later.
results = {}
# I'm using a ThreadPoolExecutor here cause it efficiently manages a pool of worker threads,
# suitable for I/O-bound tasks like API calls. This ensures parallel execution for speed.
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # I'm submitting each item categorization task to the executor here. Each task calls 'categorize_item_dynamic'
    # with the necessary arguments (item ID, description, the *shared* set, the *shared* lock, client, model).
    # I store the returned 'future' object, mapping it back to the item_id. This ensures all items are queued for processing.
    future_to_item_id = {
        executor.submit(categorize_item_dynamic, item_id, description, live_categories_set, category_lock, client, LLM_MODEL_NAME): item_id
        for item_id, description in items_to_process
    }

    processed_count = 0
    total_items = len(items_to_process)
    start_time = time.time()

    # I'm iterating through the futures as they complete here cause this allows processing results
    # as soon as they become available and provides a way to track progress. This ensures efficient handling of completed tasks.
    for future in concurrent.futures.as_completed(future_to_item_id):
        item_id = future_to_item_id[future]
        try:
            # I'm retrieving the result from the completed future here. The result is expected to be (item_id, category_name).
            # This ensures we get the output from the 'categorize_item_dynamic' function.
            _item_id_result, category = future.result() # We already have item_id, just need category
            # I'm storing the successfully retrieved category in the results dictionary using the item_id as the key.
            # This ensures the category is correctly associated with its item.
            results[item_id] = category
        except Exception as exc:
            # I'm catching potential exceptions that occurred *within the worker thread* here cause errors during
            # categorization (like persistent API failures) need to be handled.
            # This ensures that failures in individual tasks don't stop the entire process, and problematic items are marked as 'Error'.
            print(f'Error processing result for item ID {item_id}: {exc}')
            results[item_id] = "Error" # Assign 'Error' category on failure

        # --- Progress Reporting ---
        processed_count += 1
        # I'm printing progress periodically (every 100 items or at the end) here cause processing can take time,
        # and feedback assures the user the script is running. I also show processing speed and the number of categories found so far.
        # This ensures the user is informed about the progress and the dynamic nature of category discovery.
        if processed_count % 100 == 0 or processed_count == total_items:
            elapsed_time = time.time() - start_time
            items_per_sec = processed_count / elapsed_time if elapsed_time > 0 else 0
            # I'm acquiring the lock briefly to get the current count of discovered categories here cause
            # reading the size of the shared set should also be thread-safe for an accurate count.
            # This ensures the reported category count is consistent.
            with category_lock:
                current_cat_count = len(live_categories_set)
            print(f"  Processed {processed_count}/{total_items} items... ({items_per_sec:.2f} items/sec) | Categories found: {current_cat_count}")

print(f"--- Dynamic categorization complete ({processed_count} items processed) ---")

# I'm mapping the results from the dictionary back to a new 'Category' column in the DataFrame here,
# using the 'Item ID' for alignment. I'm filling any potential misses (e.g., if an ID somehow didn't get processed)
# with "Unknown". This ensures all rows get a category assigned.
df['Category'] = df['Item ID'].map(results).fillna("Unknown")


# === Step 5b: Merge Low-Count Categories into 'Other' ===
print(f"\n--- Merging Infrequent Categories (<= {LOW_COUNT_THRESHOLD} items) into 'Other' ---")

# I'm calculating the frequency of each category assigned in the previous step here cause we need counts
# to identify infrequent ones. This ensures we know how many items fall into each category.
category_counts = df['Category'].value_counts()

# I'm identifying categories whose count is at or below the threshold here, *excluding* the core categories
# 'Other', 'Unknown', 'Error' cause these should not be merged away. This ensures only non-essential,
# infrequent categories are targeted for merging.
categories_to_merge = category_counts[
    (category_counts <= LOW_COUNT_THRESHOLD) &
    (~category_counts.index.isin(['Other', 'Unknown', 'Error']))
].index.tolist()

# I'm checking if there are any categories to merge here cause if all categories are frequent enough,
# no action is needed. This ensures we only proceed if merging is necessary.
if categories_to_merge:
    print(f"Found {len(categories_to_merge)} categories to merge into 'Other'.")
    # Optionally print the list if not too long for transparency
    if len(categories_to_merge) < 50:
        print(f"Merging: {categories_to_merge}")
    else:
        print("(List of categories to merge is too long to display)")

    # I'm creating a mapping dictionary for the replacement here cause the .replace() method works efficiently with a dict.
    # This ensures easy specification of the merge operation.
    merge_map = {cat: 'Other' for cat in categories_to_merge}

    # I'm applying the merge operation using pandas' .replace() method on the 'Category' column here cause
    # it's an efficient way to replace multiple values at once. This ensures the low-count categories are updated to 'Other'.
    df['Category'] = df['Category'].replace(merge_map)
    print(f"Merged {len(categories_to_merge)} low-count categories into 'Other'.")
else:
    print("No low-count categories found to merge.")

# --- Removed Rule-Based Merging and LLM Generalization Steps ---
# These steps were part of previous iterations but are removed in this version which focuses
# on dynamic categorization followed by simple low-count merging.

# === Step 6: Final Analysis & Save Results ===
print("\n--- Final Analysis & Results ---")

# I'm getting the unique list of final categories present in the DataFrame *after* merging here cause
# this represents the final taxonomy generated by the script. This ensures we report the actual final categories.
final_categories = sorted(df['Category'].dropna().unique())
print(f"Final Categories Found ({len(final_categories)}):")
# I'm optionally printing the list of final categories if it's not excessively long here cause
# seeing the list is helpful, but huge lists clutter the output. This ensures useful feedback without overwhelming the user.
if len(final_categories) < 150:
    print(final_categories)
else:
    print(f"(List too long to display: {len(final_categories)} categories)")


print(f"\nValue Counts per Category (Post-Merging):")
# I'm printing the value counts of the final 'Category' column here cause it shows the distribution
# of items across the final categories. This ensures visibility into the categorization results.
print(df['Category'].value_counts())

# --- Optional Spending Analysis ---
# I'm searching for common column names representing spending or cost here cause analyzing spending
# per category is a common requirement. This ensures the script tries to find relevant financial data.
spending_col = None
possible_spend_cols = ['Total Bcy', 'Amount', 'Total Amount', 'Cost', 'Value', 'PO Value', 'Line Total', 'Extended Price']
for col_name in possible_spend_cols:
    if col_name in df.columns:
        spending_col = col_name
        print(f"\n--- Spending Analysis (using '{spending_col}' column) ---")
        break

if spending_col:
    try:
        # I'm attempting to convert the identified spending column to numeric here, coercing errors to NaN,
        # cause the column might contain non-numeric values or be stored as text. This ensures calculations can be performed.
        df[spending_col] = pd.to_numeric(df[spending_col], errors='coerce')
        # I'm dropping rows where either spending or category is missing here cause they cannot be included in the aggregation.
        # This ensures the analysis is based on valid, complete data points.
        df_spend = df.dropna(subset=[spending_col, 'Category'])

        if not df_spend.empty:
            # I'm grouping the DataFrame by the final 'Category' and summing the spending column here cause
            # this calculates the total spend for each category. I sort the results descending.
            # This ensures the primary analysis goal (spend per category) is achieved.
            category_spending = df_spend.groupby('Category')[spending_col].sum().sort_values(ascending=False)
            print(f"\nTotal Spending per Category ('{spending_col}'):")
            # I'm setting a display format for floats here cause large numbers are easier to read with commas.
            # This ensures better presentation of the spending figures.
            pd.options.display.float_format = '{:,.2f}'.format
            print(category_spending)
            pd.reset_option('display.float_format') # Reset format to default
        else:
            print(f"\nNo valid numeric data found in '{spending_col}' column after filtering NaNs. Skipping spending analysis.")
    except Exception as e:
        # I'm catching errors during the spending analysis here cause issues like unexpected data types can occur.
        # This ensures the script reports the problem and continues to the saving step.
        print(f"\nError during spending analysis: {e}. Skipping.")
        # I'm printing data type information if an error occurs here cause it helps diagnose issues with the spending column.
        # This ensures better debuggability.
        if spending_col in df.columns:
            print(f"Sample data types in '{spending_col}':\n{df[spending_col].apply(type).value_counts()}")
else:
    print("\nSpending column not found among likely candidates. Skipping spending analysis.")

# --- Save Results ---
try:
    # I'm defining a preferred column order here, putting key identifiers and the category first/last,
    # cause a structured output file is easier to read. This ensures a more organized CSV output.
    id_cols = [col for col in [id_col] if col in df.columns] # Use the identified (or created) id_col name
    desc_cols = [col for col in ['Item Name'] if col in df.columns]
    cat_cols = [col for col in ['Category'] if col in df.columns] # Use the final 'Category' column
    # Get remaining columns, excluding the ones already selected
    other_cols = [col for col in df.columns if col not in id_cols + desc_cols + cat_cols]
    # Combine in desired order: ID, Description, Other..., Category
    cols_to_save = id_cols + desc_cols + other_cols + cat_cols
    # Ensure no duplicates and maintain order (though unlikely here)
    cols_to_save = sorted(set(cols_to_save), key=cols_to_save.index)
    # Final check to ensure all selected columns actually exist in the dataframe
    cols_to_save = [col for col in cols_to_save if col in df.columns]

    df_to_save = df[cols_to_save]
    # I'm saving the final DataFrame with the added 'Category' column to a CSV file here.
    # I use 'utf-8-sig' encoding cause it handles a wide range of characters (including Arabic)
    # and includes a BOM (Byte Order Mark) which helps Excel open the file correctly.
    # This ensures the results are saved persistently and are easily accessible.
    df_to_save.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\nFinal categorized data saved to CSV: {OUTPUT_CSV_PATH}")
except Exception as e:
    # I'm catching potential errors during file saving here cause disk issues or permission problems can occur.
    # This ensures the script reports saving failures.
    print(f"\nError saving results to CSV: {e}")

print("\nScript finished.")
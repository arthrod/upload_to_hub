# JSONL Processor & Uploader

This script processes JSONL files in a specified directory and uploads them to the HuggingFace Hub. It offers several key features:

- **Validation of JSONL Files:**  
  Each file is checked for JSON syntax errors. If errors are found, you are prompted to choose one of these options:
  - **[p]** Proceed without the file
  - **[i]** Interrupt the process
  - **[e]** Easy Fix (default): Rewrite the file with only valid JSON lines

- **Combined vs. Individual Uploads:**  
  Use the `--shove` flag to load all JSONL files into memory, normalize columns, and upload them as one combined dataset. Otherwise, each file is processed and uploaded individually.

- **Flattening Nested JSON Objects:**  
  With the `--flatten` flag, nested JSON objects are flattened into dot notation (e.g. a nested key `ex: { "abc": 1 }` becomes `ex.abc: 1`).  
  If duplicate flattened keys would occur, they are automatically differentiated by appending a numeric suffix (e.g. `ex.abc_2`).

- **Archiving:**  
  After processing, the files are moved to an archive folder named after the generated repository suffix.

- **Logging:**  
  Detailed logging is performed using [Loguru](https://github.com/Delgan/loguru). Logs are written to `app.log` with rotation at 1 MB.

## Installation

Ensure you have Python 3.12 or newer. Then install the required packages. For example:

```bash
uv pip install typer datasets loguru

Usage

Run the script from the command line:

uv run hf_upload.py [DIRECTORY] [OPTIONS]

Arguments & Options
	•	DIRECTORY
The directory containing JSONL files (defaults to the current working directory).
	•	–repo_name, -r
Custom repository name. If omitted, a name is generated using the first 8 characters of the directory name and the current UTC date.
	•	–user_name, -u
Your HuggingFace username (default: cicero-im).
	•	–shove, -s
Load all files into memory, normalize columns, and upload as a single combined dataset.
	•	–flaten
Flatten nested JSON objects into dot notation columns. (Note the intentional misspelling to match legacy usage.)
If a dictionary contains non-dict items along with nested ones, the original key is preserved.
	•	–hf-token
Your HuggingFace token. If not provided, the script will use the HF_TOKEN environment variable.
Note: The script will abort if no token is found.

Example

python your_script.py /path/to/jsonl/files --repo_name my_dataset --user_name your_username --shove --flaten --hf-token your_hf_token

Logging

The script uses Loguru for logging. All log messages are saved to app.log in the current directory with a rotation limit of 1 MB.

Error Handling

If any JSONL file contains syntax errors, you will be prompted to choose one of the following options:
	•	[p]: Proceed without uploading the problematic file.
	•	[i]: Interrupt the process.
	•	[e]: Attempt an “easy fix” by rewriting the file with only valid lines (default).

License

This project is licensed under the MIT License.

Happy processing and uploading!

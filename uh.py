# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "typer",
#     "datasets",
#     "loguru",
# ]
# ///

import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import typer
from datasets import load_dataset
from loguru import logger

app = typer.Typer()

# Configure logger to a rotating file.
logger.add('app.log', rotation='1 MB')


def get_jsonl_files(directory: Path) -> list[Path]:
    """Return all JSONL files in the given directory."""
    return list(directory.glob('*.jsonl'))


def validate_jsonl_file(file: Path) -> tuple[bool, list[dict], list[int]]:
    """
    Validate a JSONL file.

    Returns:
      - has_error (bool): True if any JSONDecodeError occurred.
      - valid_data (list): List of valid JSON objects.
      - error_lines (list): Line numbers where errors were detected.
    """
    valid_data = []
    error_lines = []
    try:
        with file.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f, start=1):
                try:
                    record = json.loads(line)
                    valid_data.append(record)
                except json.JSONDecodeError as e:
                    logger.error(f'Error parsing line {i} in {file.name}: {e}')
                    error_lines.append(i)
    except Exception as e:
        logger.error(f'Failed to read {file.name}: {e}')
        error_lines.append(-1)
    return (len(error_lines) > 0), valid_data, error_lines


def attempt_easy_fix(file: Path, valid_data: list[dict]) -> bool:
    """
    Attempt an easy fix on a JSONL file by rewriting it with only valid JSON lines.
    Returns True if the file was fixed (and is non-empty), otherwise False.
    """
    if not valid_data:
        logger.error(f'No valid data found in {file.name} after attempting fix.')
        return False
    try:
        with file.open('w', encoding='utf-8') as f:
            for record in valid_data:
                f.write(json.dumps(record) + '\n')
        logger.info(f'Easy fix applied to {file.name}.')
        return True
    except Exception as e:
        logger.error(f'Failed to write fixed data to {file.name}: {e}')
        return False


def create_archive_directory(directory: Path, processed_files: list[Path], archive_folder: str) -> None:
    """
    Create an archive directory (named after the repository) and move the processed files into it.
    """
    archive_dir = directory / archive_folder
    try:
        archive_dir.mkdir(exist_ok=True)
        for file in processed_files:
            shutil.move(str(file), str(archive_dir / file.name))
        logger.info(f'Archived files to {archive_dir}.')
    except (OSError, shutil.Error) as e:
        logger.warning(f'Could not archive files: {e}')


def repo_exists(repo_id: str, token: str) -> bool:
    """
    Check if a dataset repository exists on Hugging Face Hub.
    Uses the API endpoint for datasets and includes the token in the Authorization header.
    """
    url = f'https://huggingface.co/api/datasets/{repo_id}'
    req = urllib.request.Request(url)
    req.add_header('Authorization', f'Bearer {token}')
    try:
        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        else:
            logger.error(f'HTTPError while checking repo: {e}')
            return True
    except Exception as e:
        logger.error(f'Error checking repo existence: {e}')
        return True
    return False


@app.command()
def process_jsonl(
    directory: Path = typer.Argument(Path.cwd(), help='Directory containing JSONL files'),
    repo_name: str | None = typer.Option(
        None, '--repo_name', '-r', help="Custom repository name. If not provided, uses the first file's name."
    ),
    user_name: str = typer.Option('cicero-im', '--user_name', '-u', help='HuggingFace username'),
    compile: bool = typer.Option(
        True,
        '--compile',
        help='Compile all JSONL files into a single dataset and push once (default True). If False, each file is pushed as its own dataset.',
    ),
    flatten: bool = typer.Option(
        False, '--flatten', help='Flatten nested JSON objects using datasets.flatten(max_depth=16)'
    ),
    private: bool | None = typer.Option(
        None,
        '--private',
        help="Whether to make the repo private. If None (default), the repo will be public unless the organization's default is private.",
    ),
    create_pr: bool = typer.Option(
        False, '--create-pr', help='Create a PR with the uploaded files instead of directly committing'
    ),
    hf_token: str | None = typer.Option(
        None, '--hf-token', help='HuggingFace token (defaults to the HF_TOKEN env variable)'
    ),
) -> None:
    """
    Process JSONL files and upload to HuggingFace Hub.

    The script validates each JSONL file and, if errors are found, prompts whether to proceed, interrupt, or attempt an easy fix.

    - When **--compile** is True (default):
      All JSONL files are normalized and compiled into a single dataset. The repository (and archive folder) is named based on the first file's name (if no custom repo name is provided). **push_to_hub** is then called once.

    - When **--compile** is False:
      Each valid JSONL file is processed separately and pushed as its own dataset. The repository for each file is named using the file's stem (or the provided repo_name plus an underscore and the file's stem).

    In both cases, if the repository already exists on Hugging Face, you are prompted to choose one of:
      - **[a]ppend (default):** Append a numeric suffix to the repo name.
      - **[o]verwrite:** Overwrite the existing repository.
      - **[p]r:** Create a pull request with the uploaded files.

    The HuggingFace token is passed to both load_dataset and push_to_hub.
    """
    token = hf_token if hf_token is not None else os.environ.get('HF_TOKEN')
    if token is None:
        typer.echo(
            typer.style(
                'Error: HF_TOKEN is not set in the environment and no --hf-token provided.', fg='red', bold=True
            )
        )
        raise typer.Exit(1)

    files = get_jsonl_files(directory)
    if not files:
        typer.echo(typer.style('No JSONL files found in the directory.', fg='red', bold=True))
        raise typer.Exit(1)

    valid_files = []
    for file in files:
        has_error, valid_data, error_lines = validate_jsonl_file(file)
        if has_error:
            msg = (
                f"File '{file.name}' has syntax errors on lines: {error_lines}.\n"
                "Choose an option: [p]roceed without this file, [i]nterrupt, [e]asy fix (default 'e')"
            )
            choice = typer.prompt(msg, default='e')
            if choice.lower() == 'i':
                logger.error(f'User chose to interrupt due to errors in {file.name}.')
                raise typer.Exit(1)
            elif choice.lower() == 'p':
                logger.info(f'User chose to proceed without file {file.name}.')
                continue
            else:
                if attempt_easy_fix(file, valid_data):
                    logger.info(f'File {file.name} fixed and validated.')
                else:
                    logger.error(f'Easy fix failed for {file.name}. Skipping file.')
                    continue
        valid_files.append(file)

    if not valid_files:
        typer.echo(typer.style('No valid JSONL files to process after validation.', fg='red', bold=True))
        raise typer.Exit(1)

    if compile:
        # For compiling, use one repo.
        first_file = valid_files[0]
        if repo_name is None:
            repo_name = first_file.stem
        archive_folder = repo_name
        repo_id = f'{user_name}/{repo_name}'

        if repo_exists(repo_id, token):
            choice = typer.prompt(
                typer.style(
                    f"Repository '{repo_id}' already exists. Choose an action: [a]ppend (default), [o]verwrite, [p]r",
                    fg='yellow',
                    bold=True,
                ),
                default='a',
            )
            if choice.lower() == 'a':
                base = repo_id
                counter = 1
                new_repo_id = f'{base}_{counter}'
                while repo_exists(new_repo_id, token):
                    counter += 1
                    new_repo_id = f'{base}_{counter}'
                repo_id = new_repo_id
                typer.echo(typer.style(f'New repository id set to: {repo_id}', fg='blue', bold=True))
            elif choice.lower() == 'o':
                typer.echo(typer.style(f"Overwriting repository '{repo_id}'.", fg='red', bold=True))
            elif choice.lower() == 'p':
                create_pr = True
                typer.echo(typer.style(f"Will create a PR for repository '{repo_id}'.", fg='blue', bold=True))
            else:
                typer.echo(typer.style('Invalid option. Aborting.', fg='red', bold=True))
                raise typer.Exit(1)

        combined_dataset = load_dataset('json', data_files=[str(file) for file in valid_files], token=token)['train']
        if flatten:
            combined_dataset = combined_dataset.flatten(max_depth=16)
            # Compute the new flattened columns compared to the union of all columns.
            flattened_columns = set(combined_dataset.column_names) - set().union(*[
                set(load_dataset('json', data_files=str(file), token=token)['train'].column_names)
                for file in valid_files
            ])
            typer.echo(typer.style(f'Flattened columns added: {sorted(flattened_columns)}', fg='blue', bold=True))
        typer.echo(typer.style('Uploading compiled dataset...', fg='magenta', bold=True))
        combined_dataset.push_to_hub(repo_id, token=token, private=private, create_pr=create_pr)
    else:
        # Process each file separately.
        for file in valid_files:
            file_stem = file.stem
            if repo_name is not None:
                local_repo_name = f'{repo_name}_{file_stem}'
            else:
                local_repo_name = file_stem
            local_repo_id = f'{user_name}/{local_repo_name}'

            if repo_exists(local_repo_id, token):
                choice = typer.prompt(
                    typer.style(
                        f"Repository '{local_repo_id}' already exists. Choose an action: [a]ppend (default), [o]verwrite, [p]r",
                        fg='yellow',
                        bold=True,
                    ),
                    default='a',
                )
                if choice.lower() == 'a':
                    base = local_repo_id
                    counter = 1
                    new_repo_id = f'{base}_{counter}'
                    while repo_exists(new_repo_id, token):
                        counter += 1
                        new_repo_id = f'{base}_{counter}'
                    local_repo_id = new_repo_id
                    typer.echo(
                        typer.style(f'New repository id for {file.name} set to: {local_repo_id}', fg='blue', bold=True)
                    )
                elif choice.lower() == 'o':
                    typer.echo(
                        typer.style(f"Overwriting repository '{local_repo_id}' for {file.name}.", fg='red', bold=True)
                    )
                elif choice.lower() == 'p':
                    create_pr = True
                    typer.echo(
                        typer.style(
                            f"Will create a PR for repository '{local_repo_id}' for {file.name}.", fg='blue', bold=True
                        )
                    )
                else:
                    typer.echo(typer.style('Invalid option. Aborting.', fg='red', bold=True))
                    raise typer.Exit(1)

            ds = load_dataset('json', data_files=str(file), token=token)['train']
            original_columns = set(ds.column_names)
            if flatten:
                ds = ds.flatten(max_depth=16)
                flattened_columns = set(ds.column_names) - original_columns
                typer.echo(
                    typer.style(
                        f'For file {file.name}, flattened columns added: {sorted(flattened_columns)}',
                        fg='blue',
                        bold=True,
                    )
                )
            typer.echo(
                typer.style(
                    f'Processing and uploading {file.name} with columns: {sorted(ds.column_names)}...',
                    fg='magenta',
                    bold=True,
                )
            )
            ds.push_to_hub(local_repo_id, token=token, private=private, create_pr=create_pr)
        archive_folder = repo_name if repo_name is not None else 'individual'

    create_archive_directory(directory, valid_files, archive_folder)
    typer.echo(typer.style('Upload complete and files archived.', fg='cyan', bold=True))


if __name__ == '__main__':
    app()

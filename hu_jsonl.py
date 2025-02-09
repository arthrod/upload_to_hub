import json
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import typer
from datasets import concatenate_datasets, load_dataset
from loguru import logger

app = typer.Typer()

# Configure logger to a rotating file.
logger.add('app.log', rotation='1 MB')


def get_jsonl_files(directory: Path | str) -> list[Path]:
    """Return all JSONL files in the given directory."""
    directory = Path(directory) if isinstance(directory, str) else directory
    if not directory.exists():
        logger.error(f"Directory doesn't exist: {directory}")
        return []
    if not directory.is_dir():
        logger.error(f'Path is not a directory: {directory}')
        return []
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
    Create an archive directory and move the processed files into it.
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
    """Check if a dataset repository exists on Hugging Face Hub."""
    url = f'https://huggingface.co/api/datasets/{repo_id}'
    req = urllib.request.Request(url)
    req.add_header('Authorization', f'Bearer {token}')
    try:
        with urllib.request.urlopen(req) as response:
            return response.getcode() == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        logger.error(f'HTTPError while checking repo: {e}')
        return True
    except Exception as e:
        logger.error(f'Error checking repo existence: {e}')
        return True


def load_and_process_dataset(file: Path, token: str, flatten: bool = False) -> tuple[set, any]:
    """Load a dataset file and return its columns and dataset object."""
    ds = load_dataset('json', data_files=str(file), token=token)['train']
    original_columns = set(ds.column_names)
    if flatten:
        ds = ds.flatten(max_depth=16)
    return original_columns, ds


def process_jsonl_files(
    directory: Path,
    repo_name: str | None,
    user_name: str,
    consolidate: bool,
    flatten: bool,
    private: bool | None,
    create_pr: bool,
    token: str,
) -> None:
    """Process JSONL files and upload to HuggingFace Hub."""
    try:
        work_dir = directory.resolve()
        if not work_dir.exists():
            typer.echo(typer.style(f'Directory does not exist: {work_dir}', fg='red', bold=True))
            raise typer.Exit(1)
        if not work_dir.is_dir():
            typer.echo(typer.style(f'Path is not a directory: {work_dir}', fg='red', bold=True))
            raise typer.Exit(1)

        files = get_jsonl_files(work_dir)
        if not files:
            typer.echo(typer.style(f'No JSONL files found in the directory: {work_dir}', fg='red', bold=True))
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(typer.style(f'Error accessing directory: {e}', fg='red', bold=True))
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
            if choice.lower() == 'p':
                logger.info(f'User chose to proceed without file {file.name}.')
                continue
            if attempt_easy_fix(file, valid_data):
                logger.info(f'File {file.name} fixed and validated.')
            else:
                logger.error(f'Easy fix failed for {file.name}. Skipping file.')
                continue
        valid_files.append(file)

    if not valid_files:
        typer.echo(typer.style('No valid JSONL files to process after validation.', fg='red', bold=True))
        raise typer.Exit(1)

    if consolidate:
        # Load all datasets first to get complete column information
        datasets_list = []
        all_columns = set()
        for file in valid_files:
            _, ds = load_and_process_dataset(file, token, flatten)
            all_columns.update(ds.column_names)
            datasets_list.append(ds)

        # Add missing columns with None values to each dataset
        for i, ds in enumerate(datasets_list):
            missing_columns = all_columns - set(ds.column_names)
            if missing_columns:
                for col in missing_columns:
                    ds = ds.add_column(col, [None] * len(ds))
                datasets_list[i] = ds

        # Concatenate all datasets
        combined_dataset = concatenate_datasets(datasets_list)

        # Set up repository name and handle existing repos
        if repo_name is None:
            repo_name = valid_files[0].stem
        repo_id = f'{user_name}/{repo_name}'

        if repo_exists(repo_id, token):
            choice = typer.prompt(
                f"Repository '{repo_id}' already exists. Choose: [a]ppend (default), [o]verwrite, [p]roceed without this file`", default='a'
            )
            if choice.lower() == 'a':
                counter = 1
                while repo_exists(f'{repo_id}_{counter}', token):
                    counter += 1
                repo_id = f'{repo_id}_{counter}'
            elif choice.lower() == 'p':
                create_pr = True
            elif choice.lower() != 'o':
                raise typer.Exit(1)

        combined_dataset.push_to_hub(repo_id, token=token, private=private, create_pr=create_pr)

    else:
        # Process each file individually
        for file in valid_files:
            local_repo_name = f'{repo_name}_{file.stem}' if repo_name else file.stem
            local_repo_id = f'{user_name}/{local_repo_name}'

            _, ds = load_and_process_dataset(file, token, flatten)

            if repo_exists(local_repo_id, token):
                choice = typer.prompt(
                    f"Repository '{local_repo_id}' exists. Choose: [a]ppend (default), [o]verwrite, [p]roceed without this file",
                    default='a',
                )
                if choice.lower() == 'a':
                    counter = 1
                    while repo_exists(f'{local_repo_id}_{counter}', token):
                        counter += 1
                    local_repo_id = f'{local_repo_id}_{counter}'
                elif choice.lower() == 'p':
                    create_pr = True
                elif choice.lower() != 'o':
                    continue

            typer.echo(f'Uploading {file.name} to {local_repo_id}...')
            ds.push_to_hub(local_repo_id, token=token, private=private, create_pr=create_pr)

    # Archive processed files
    archive_folder = f'archive_{repo_name}' if repo_name else 'archive_individual'
    create_archive_directory(directory, valid_files, archive_folder)
    typer.echo(typer.style('Upload complete and files archived.', fg='green', bold=True))


@app.command()
def main(
    directory: str = typer.Option('.', '--directory', '-d', help='Directory containing JSONL files'),
    repo_name: str | None = typer.Option(None, '--repo-name', '-r', help='Custom repository name'),
    user_name: str = typer.Option('cicero-im', '--user-name', '-u', help='HuggingFace username'),
    consolidate: bool = typer.Option(True, '--consolidate/--no-consolidate', help='Consolidate all files into a single dataset'),
    flatten: bool = typer.Option(False, '--flatten/--no-flatten', help='Flatten nested JSON objects'),
    private: bool | None = typer.Option(None, '--private', help='Make repository private'),
    create_pr: bool = typer.Option(False, '--create-pr', help='Create PR instead of direct commit'),
    hf_token: str | None = typer.Option(None, '--hf-token', help='HuggingFace token'),
) -> None:
    """Process JSONL files and upload to HuggingFace Hub."""
    token = hf_token if hf_token is not None else os.environ.get('HF_TOKEN')
    if token is None:
        typer.echo(typer.style('Error: HF_TOKEN is not set in the environment and no --hf-token provided.', fg='red', bold=True))
        raise typer.Exit(1)

    process_jsonl_files(
        directory=Path(directory),
        repo_name=repo_name,
        user_name=user_name,
        consolidate=consolidate,
        flatten=flatten,
        private=private,
        create_pr=create_pr,
        token=token,
    )
    """Process JSONL files and upload to HuggingFace Hub."""
    token = hf_token if hf_token is not None else os.environ.get('HF_TOKEN')
    if token is None:
        typer.echo(typer.style('Error: HF_TOKEN is not set in the environment and no --hf-token provided.', fg='red', bold=True))
        raise typer.Exit(1)

    try:
        # Use current directory if no directory is specified
        work_dir = Path(directory if directory is not None else '.').resolve()
        if not work_dir.exists():
            typer.echo(typer.style(f'Directory does not exist: {work_dir}', fg='red', bold=True))
            raise typer.Exit(1)
        if not work_dir.is_dir():
            typer.echo(typer.style(f'Path is not a directory: {work_dir}', fg='red', bold=True))
            raise typer.Exit(1)

        files = get_jsonl_files(work_dir)
        if not files:
            typer.echo(typer.style(f'No JSONL files found in the directory: {work_dir}', fg='red', bold=True))
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(typer.style(f'Error accessing directory: {e}', fg='red', bold=True))
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
            if choice.lower() == 'p':
                logger.info(f'User chose to proceed without file {file.name}.')
                continue
            if attempt_easy_fix(file, valid_data):
                logger.info(f'File {file.name} fixed and validated.')
            else:
                logger.error(f'Easy fix failed for {file.name}. Skipping file.')
                continue
        valid_files.append(file)

    if not valid_files:
        typer.echo(typer.style('No valid JSONL files to process after validation.', fg='red', bold=True))
        raise typer.Exit(1)

    if consolidate:
        # Load all datasets first to get complete column information
        datasets_list = []
        all_columns = set()
        for file in valid_files:
            _, ds = load_and_process_dataset(file, token, flatten)
            all_columns.update(ds.column_names)
            datasets_list.append(ds)

        # Add missing columns with None values to each dataset
        for i, ds in enumerate(datasets_list):
            missing_columns = all_columns - set(ds.column_names)
            if missing_columns:
                for col in missing_columns:
                    ds = ds.add_column(col, [None] * len(ds))
                datasets_list[i] = ds

        # Concatenate all datasets
        combined_dataset = concatenate_datasets(datasets_list)

        # Set up repository name and handle existing repos
        if repo_name is None:
            repo_name = valid_files[0].stem
        repo_id = f'{user_name}/{repo_name}'

        if repo_exists(repo_id, token):
            choice = typer.prompt(
                f"Repository '{repo_id}' already exists. Choose: [a]ppend (default), [o]verwrite, [p]roceed without this file",
                default='a',
            )
            if choice.lower() == 'a':
                counter = 1
                while repo_exists(f'{repo_id}_{counter}', token):
                    counter += 1
                repo_id = f'{repo_id}_{counter}'
            elif choice.lower() == 'p':
                create_pr = True
            elif choice.lower() != 'o':
                raise typer.Exit(1)

        combined_dataset.push_to_hub(repo_id, token=token, private=private, create_pr=create_pr)

    else:
        # Process each file individually
        for file in valid_files:
            local_repo_name = f'{repo_name}_{file.stem}' if repo_name else file.stem
            local_repo_id = f'{user_name}/{local_repo_name}'

            _, ds = load_and_process_dataset(file, token, flatten)

            if repo_exists(local_repo_id, token):
                choice = typer.prompt(f"Repository '{local_repo_id}' exists. Choose: [a]ppend (default), [o]verwrite, [p]r", default='a')
                if choice.lower() == 'a':
                    counter = 1
                    while repo_exists(f'{local_repo_id}_{counter}', token):
                        counter += 1
                    local_repo_id = f'{local_repo_id}_{counter}'
                elif choice.lower() == 'p':
                    create_pr = True
                elif choice.lower() != 'o':
                    continue

            typer.echo(f'Uploading {file.name} to {local_repo_id}...')
            ds.push_to_hub(local_repo_id, token=token, private=private, create_pr=create_pr)

    # Archive processed files
    archive_folder = f'archive_{repo_name}' if repo_name else 'archive_individual'
    create_archive_directory(directory, valid_files, archive_folder)
    typer.echo(typer.style('Upload complete and files archived.', fg='green', bold=True))


if __name__ == '__main__':
    app()

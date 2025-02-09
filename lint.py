import subprocess
import sys
from pathlib import Path


def run_linting():
    """
    Run comprehensive code linting and formatting checks.
    Returns True if all checks pass, False otherwise.
    """
    print("Starting code quality checks...")

    # Define the files to check (excluding virtual environments).
    python_files = [
        str(f)
        for f in Path(".").rglob("*.py")
        if "venv" not in str(f) and "env" not in str(f)
    ]

    success = True

    # Run Black formatter check.
    print("\n1. Running Black formatter...")
    try:
        subprocess.run(["black", "--check"] + python_files, check=True)
        print("✓ Black formatting check passed")
    except subprocess.CalledProcessError:
        success = False
        print("× Black formatting check failed")
        print("   To fix: Run 'black .' to format your code")

    # Run Flake8 linter.
    print("\n2. Running Flake8 linter...")
    try:
        subprocess.run(["flake8"] + python_files, check=True)
        print("✓ Flake8 check passed")
    except subprocess.CalledProcessError:
        success = False
        print("× Flake8 check failed")

    # Print summary.
    print("\nLinting Summary:")
    if success:
        print("All checks passed! ✨ Code is ready for review.")
    else:
        print("Some checks failed. Please fix the issues above.")

    return success


if __name__ == "__main__":
    sys.exit(not run_linting())

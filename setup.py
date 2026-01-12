"""Setup script for LLM-for-OR package."""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="llm-for-or",
        packages=find_packages(where="."),
        package_dir={"": "."},
    )

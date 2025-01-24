from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="autobyteus_llm_client",
    version="0.1.0",
    author="Ryan Zheng",
    author_email="ryan.zheng.work@gmail.com",
    description="Async Python client for Autobyteus LLM API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/autobyteus-llm-client",
    packages=find_packages(include=["autobyteus_llm_client", "autobyteus_llm_client.*"]),
    python_requires=">=3.8",
    install_requires=[
        "httpx",
    ],
    extras_require={
        "test": [
            "pytest-asyncio",
            "python-dotenv"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="llm client async",
)

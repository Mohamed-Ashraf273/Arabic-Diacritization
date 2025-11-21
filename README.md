# Arabic Diacritization

A machine learning project for automatic Arabic text diacritization (تشكيل النصوص العربية).

## Overview

This project aims to automatically add diacritical marks (tashkeel) to Arabic text, which is essential for proper pronunciation and meaning disambiguation in the Arabic language.

## Project Structure

```
Arabic-Diacritization/
├── README.md
├── features/
│   └── feature_extractor.py    # Feature extraction implementations
├── models/
│   └── model.py                # Model implementations
└── preprocessing/
    └── preprocessor.py         # Text preprocessing implementations
```

## For Developers

When implementing new components, please follow these guidelines:

- **Models**: Inherit from the base model class and implement all required methods
- **Preprocessors**: Inherit from the base preprocessor class and implement all required methods
- **Feature Extractors**: Inherit from the base feature extractor class and implement all required methods

This ensures consistency and maintainability across the codebase.

## Getting Started

### Prerequisites

- Python 3.7+
- Required dependencies (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/Mohamed-Ashraf273/Arabic-Diacritization.git
cd Arabic-Diacritization

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Example usage will be added here
```
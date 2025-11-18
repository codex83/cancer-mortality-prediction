# Contributing to Cancer Mortality Prediction

First off, thank you for considering contributing to this project! It's people like you that make this project better.

## Code of Conduct

This project and everyone participating in it is governed by respect and professionalism. By participating, you are expected to uphold this standard.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the behavior you observed and what you expected**
- **Include screenshots if relevant**
- **Include your environment details** (OS, Python version, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the proposed enhancement**
- **Explain why this enhancement would be useful**
- **List any similar features in other projects if applicable**

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code, add tests if applicable
3. Ensure your code follows the existing style
4. Update documentation as needed
5. Write a clear commit message

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/cancer-mortality-prediction.git

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create a new branch
git checkout -b feature/your-feature-name
```

## Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex logic

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally

## Project Structure

- `pyfiles/` - Core Python modules
- `data/` - Dataset files
- `models/` - Saved model files (generated)
- `predictions/` - Prediction outputs (generated)
- `evidently_reports/` - Monitoring reports (generated)
- `cancer_mortality_prediction.ipynb` - Main Jupyter notebook

## Testing

Before submitting a pull request, ensure:

- Your code runs without errors
- The notebook executes from top to bottom
- All tests pass (if applicable)
- Documentation is updated

## Questions?

Feel free to open an issue with your question or reach out directly.

Thank you for your contributions! 🎉


# Contributing to AI-Powered HR Document Intelligence Platform

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Guidelines](#development-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Questions and Support](#questions-and-support)

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code. Please report unacceptable behavior to dino.lozina@live.com.

## Getting Started

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Git
- Basic understanding of microservices architecture

### Setting Up Development Environment

1. **Fork the repository**
   
   First, fork the repository on GitHub:
   - Go to https://github.com/dlozina/ai-document-platform
   - Click the "Fork" button in the top-right corner
   - This creates a copy of the repository in your GitHub account

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/ai-document-platform.git
   cd ai-document-platform
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/dlozina/ai-document-platform.git
   ```

4. **Start the development environment**
   ```bash
   cd backend
   docker compose up -d
   ```

5. **Verify installation**
   ```bash
   # Check all services are running
   docker compose ps
   
   # Test API endpoints
   curl http://localhost:8003/health
   ```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes** - Fix issues and improve stability
- **Feature enhancements** - Add new functionality
- **Documentation** - Improve docs, examples, and guides
- **Testing** - Add tests or improve test coverage
- **Performance** - Optimize existing code
- **UI/UX** - Improve user experience

### Before You Start

1. **Check existing issues** - Look for open issues that match your interests
2. **Create an issue** - For significant changes, discuss your approach first
3. **Fork the repository** - Create your own fork to work on

### Fork Workflow

When contributing, follow this workflow:

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** - Write code, tests, and documentation

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill out the PR template
   - Submit the PR

6. **Keep your fork updated**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main
   ```

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8 guidelines
- **Documentation**: Use clear, concise language
- **Comments**: Explain complex logic and business decisions
- **Naming**: Use descriptive variable and function names

### Testing

- Write tests for new functionality
- Ensure existing tests pass
- Aim for meaningful test coverage
- Test both success and error scenarios

### Documentation

- Update relevant documentation when adding features
- Include examples for new APIs
- Keep README files current
- Document any breaking changes

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add semantic search to query service
fix: resolve OCR processing timeout issue
docs: update API documentation for new endpoints
test: add unit tests for embedding processor
```

## Pull Request Process

### Before Submitting

1. **Test your changes** - Ensure all tests pass
2. **Update documentation** - Include relevant doc updates
3. **Check code style** - Follow project conventions
4. **Rebase if needed** - Keep a clean commit history

### Pull Request Template

When creating a PR, please include:

- **Description**: What changes you made and why
- **Type**: Bug fix, feature, documentation, etc.
- **Testing**: How you tested your changes
- **Breaking changes**: Any breaking changes (if applicable)
- **Related issues**: Link to relevant issues

### Review Process

1. **Automated checks** - CI/CD pipeline runs automatically
2. **Code review** - Maintainer reviews the code
3. **Feedback** - Address any requested changes
4. **Approval** - Maintainer approves and merges

### Response Time

- Initial review: Within 3-5 business days
- Follow-up reviews: Within 1-2 business days
- Questions: Usually within 24 hours

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, Docker version
- **Steps to reproduce**: Clear, numbered steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs**: Relevant error messages or logs
- **Screenshots**: If applicable

### Feature Requests

For feature requests, please include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches you considered
- **Additional context**: Any other relevant information

## Questions and Support

### Getting Help

- **GitHub Discussions**: For general questions and ideas
- **Issues**: For bug reports and feature requests
- **Email**: dino.lozina@live.com for direct contact

### Community Guidelines

- Be respectful and constructive
- Help others learn and grow
- Share knowledge and best practices
- Follow the Code of Conduct

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to make HR processes more efficient and intelligent! 🚀

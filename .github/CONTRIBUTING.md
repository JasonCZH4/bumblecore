# Contributing to BumbleCore

Thank you for your interest in BumbleCore! We welcome contributions of all kinds.

Whether it's submitting code, improving documentation, answering questions, or sharing your experience, every contribution you make helps make this project better. If BumbleCore has been helpful to you, feel free to give the project a ⭐️ or share it with others.

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) when participating and be respectful and kind to everyone.

## How You Can Contribute

Here are some ways to get involved:

* **Fix Bugs** - Check out tagged issues in [Issues](https://github.com/wxhcore/bumblecore/issues)
* **Add Features** - Implement features you think would be valuable
* **Improve Documentation** - Make the docs clearer and easier to understand
* **Share Examples** - Contribute interesting use cases

## Submitting Code

**1. Fork and Clone the Repository**

1. [Fork the repository](https://github.com/wxhcore/bumblecore/fork) by clicking the Fork button on the repository page. This creates a copy of the code under your GitHub account.

2. Clone your fork to your local disk and add the base repository as a remote:

```bash
git clone git@github.com:[your-username]/bumblecore.git
cd bumblecore
git remote add upstream https://github.com/wxhcore/bumblecore.git
```

**2. Create a Development Branch**

```bash
git checkout -b feature/your-feature-name
```

**3. Set Up Development Environment**

We recommend installing in editable mode within a virtual environment:

```bash
pip install -e .
```

> 💡 If you've installed it before, uninstall it first with `pip uninstall bumblecore`

**4. Push to Your Fork**

```bash
git add .
git commit -m "Brief description of your changes"
git push origin feature/your-feature-name
```

**5. Create a Pull Request**

On GitHub, create a Pull Request from your branch to the [main repository](https://github.com/wxhcore/bumblecore). Please include in your PR description:
- The purpose and background of the changes
- Main modifications
- Related Issues (if any)

## Reporting Issues

Found a bug? Please submit it in [Issues](https://github.com/wxhcore/bumblecore/issues) and include as much as possible:

- **Reproduction Steps**: How to trigger the issue
- **Expected vs Actual**: What should happen vs what actually happens
- **Environment**: Python version, PyTorch version, CUDA version, OS, etc.
- **Error Messages**: Complete error logs or stack traces

The more detailed information you provide, the faster we can identify and resolve the issue.

## Feature Requests

Want a new feature? We welcome Feature Requests! Before submitting, please search to see if someone has made a similar suggestion.

In your description, please include:
- **Use Case**: What problem does this feature solve
- **Expected Behavior**: How you expect it to work
- **Implementation Ideas**: If you have specific thoughts, we can discuss them together

---

## License

All contributions will follow the project's [Apache License 2.0](../LICENSE). By submitting code, you agree to release your contribution under this license.

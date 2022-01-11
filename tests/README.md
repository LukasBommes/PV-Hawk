### Run Tests

Run test suitewithin Docker container in `/pvextractor` directory with the command
```
python -m unittest tests/test_*.py
```

### Coverage Report

Make sure coverage is installed by running
```
python -m pip install coverage
```

Report coverage with
```
coverage run --source=. --branch -m unittest test_quadrilaterals.py
```

# PV-Hawk Tests

This directory contains tests for the PV-Hawk computer vision pipeline.

## Test Structure

- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for full pipeline workflows

## Running Tests

### Unit Tests
```bash
python -m unittest tests/unit/test_*.py
```

### Integration Tests
```bash
python -m unittest tests/integration/test_*.py
```

### All Tests
```bash
python -m unittest tests/**/test_*.py
```

## Test Categories

- **Geometric algorithms** - Quadrilateral detection and processing
- **GPS utilities** - Coordinate transformations and interpolation  
- **Image processing** - Cropping, preprocessing, and common utilities
- **Pipeline components** - Tracking, segmentation integration

## Requirements

Install test dependencies:
```bash
pip install deepdiff hypothesis[numpy]
```

## Coverage

Generate coverage reports:
```bash
pip install coverage
coverage run --source=. --branch -m unittest tests/integration/test_*.py
coverage report -m
coverage html
```

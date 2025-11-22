# Running the Test Script

## From the project root (D:\matth\def-acc-hackathon):

```bash
python def-acc-logiq/evaluation/test_evaluation.py
```

## From the def-acc-logiq directory:

```bash
cd def-acc-logiq
python evaluation/test_evaluation.py
```

## From the evaluation directory:

```bash
cd def-acc-logiq/evaluation
python test_evaluation.py
```

## Troubleshooting

If you get a `ModuleNotFoundError: No module named 'evaluation'`:
- Make sure you're running from the correct directory
- The script should automatically add the correct path, but if it doesn't work, try running from `def-acc-logiq` directory

If you get a `ModuleNotFoundError: No module named 'pandas'`:
- This is expected for basic testing - the script will use a fallback implementation
- To use full functionality, install dependencies: `pip install -r requirements.txt`


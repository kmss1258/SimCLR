# activate pip / conda environment first

# Train SimCLR_custom model
python main.py

# Train linear model and run test
python -m testing.logistic_regression \
    with \
    model_path=./logs/0
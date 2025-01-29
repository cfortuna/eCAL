# eCAL
Simulator for the eCAL metric


## Installation
To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```

in case if you want to run LLMs you might need additional dependencies, you can install them by running the following command:
```bash
pip install transformers sentencepiece tiktoken
```
## Usage
To run the calculator, use the following command:
```bash
python run_calculator.py
```
## Configuration
The configuration is done in the `calculator_config.py` file. What specific configuration options are available can be found in the file itself.
To change the Control and Data plane overheads of the transmission layer or implement new protocols you can change the values  in the `calculators/protocol_configs.py` file.

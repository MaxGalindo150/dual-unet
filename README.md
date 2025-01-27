# Steps to run the project

1. Clone the repository

2. Create a virtual environment
```bash
python3 -m venv env
source env/bin/activate
```

3. Install the dependencies
```bash
pip install -r requirements.txt
```

3. Generate simulation data
```bash
python python generate_simulated_data.py --num_samples 5000
```

4. Train the model
```bash
python train.py --model_name attention_unet --num_epochs 200
```

5. Evaluate the model
```bash
python test.py --model_name attention_unet
```


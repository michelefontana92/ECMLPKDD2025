# FairLab

`FairLab` is a methodology that...

---

## **Installation**

We recommend setting up a new Conda environment with **Python >= 3.9**.

### **1. Create a Conda environment**
```bash
conda create -n "fairlab" python==3.9
```

### **2. Activate the environment**
```bash
conda activate fairlab
```

### **3. Clone the repository**
```bash
git clone https://anonymous.4open.science/r/ECMLPKDD2025-016B/
```

### **4. Navigate to the project directory**
```bash
cd FairLab
```

### **5. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## **Project Structure**
The project follows this structure:

```bash
.
├── LICENSE
├── README.md
├── data
│   ├── Folktables
│   ├── Compas
│   └── MEP
├── requirements.txt
└── src
    ├── architectures
    ├── callbacks
    ├── dataloaders
    ├── loggers
    ├── main.py
    ├── metrics
    ├── runs
    ├── surrogates
    └── wrappers
```

- **`data/`** → Contains the datasets used in experiments.
- **`src/`** → Contains the core implementation of FairLab.

---

## **Usage**
To run an experiment with `FairLab`:

### **1. Navigate to the `src` directory**
```bash
cd src
```

### **2. Execute `main.py` with options**
```bash
python main.py --options
```

### **Available Options**
```bash
Options:
  -r, --run TEXT              Name of the run to execute
  -p, --project_name TEXT     Name of the WandB project
  -ml, --metrics_list TEXT    List of fairness metrics
  -gl, --groups_list TEXT     List of sensitive groups
  -tl, --threshold_list FLOAT Threshold values for fairness constraints
  -d, --delta FLOAT      Value of the proximity threshold (default = 0.02)
  -ds --delta_step FLOAT Value of delta_step parameter (default = 0.01)
  -dt --delta_tol FLOAT   Value of the tolerance threshold (default = 0.05)
  -pb --performance_budget  FLOAT Value of the performance budget (default = 0.05)
```

#### **Predefined Runs (`runs` folder)**
- **`folk_fairlab`** → Uses the **FolkTables** dataset.
- **`compas_fairlab`** → Uses the **Compas** dataset.
- **`meps_fairlab`** → Uses the **MEPS** dataset.

The code supports the following **fairness metrics**:
- `demographic_parity`
- `equal_opportunity`
- `predictive_equality`
- `equalized_odds`

---

## **Logging**
The code uses **Weights & Biases (WandB)** for experiment tracking. 
To use it:
1. Create a free account at [WandB](https://wandb.ai/site/).
2. Follow the login instructions after executing the code.

We plan to allow users to choose other logging systems in the future.

---

## **Examples**

### **1. One Fairness Constraint**
Create a federation with 10 clients enforcing **Demographic Parity (DP ≤ 0.20) on GenderRace** using the **Compas** dataset, with a performance budget of 0.05 and a delta of 0.02.
```bash
python main.py -r compas_fairlab -ml demographic_parity -tl 0.20 -gl GenderRace -p Compas_GenderRace -pb 0.05 -d 0.02
```

### **2. Mixed Fairness Metrics**
Create an experiment enforcing **DP ≤ 0.20 on GenderRace** and **DP ≤ 0.20 on  on GenderAge**, using the **Compas** dataset.

```bash
python main.py -r compas_fairlab -ml demographic_parity -tl 0.20 -gl GenderRace -ml demographic_parity -tl 0.20 -gl GenderAge -p Compas_Mixed -pb 0.05 -d 0.02
```

---

## **License**
This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.

---

## **Contact**
For any questions or suggestions, feel free to reach out:


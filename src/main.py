import subprocess
import random


def run_script(script_name, random_state):
    try:
        result = subprocess.run(
            ["python", script_name, "--random_state", str(random_state)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Output of {script_name.split('/')[-1]}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}:\n{e.stderr}")

if __name__ == "__main__":
    scripts = ["/home/guoyou/OutlierDetection/TSB-UAD/example/notebooks/AnomalyD_Comp.py", "/home/guoyou/OutlierDetection/TSB-UAD/example/notebooks/AnomalyD_Orig.py", ]
    random.seed(20250412)
    random_state = random.randint(1, 4294967295)
    with open("OutlierDetection/TSB-UAD/example/results.txt", "w") as f:
        pass 
    for script in scripts:
        print(f"Running {script.split('/')[-1]}...")
        run_script(script, random_state= random_state)
        print(f"Finished running {script.split('/')[-1]}.\n")

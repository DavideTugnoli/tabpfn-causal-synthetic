import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os

# Get available CPU cores dynamically
NUM_CPUS = os.cpu_count()
NUM_WORKERS = max(1, NUM_CPUS // 2)  # Use half the available CPUs

print(f"Optimizing for {NUM_CPUS} CPU cores, running {NUM_WORKERS} tasks in parallel.")

# List of tasks. Each task is a list where the first element is the script, and the following elements are the arguments for that script.
tasks = []
for seed in [8,9,101,78,61,2042,732,25,44,87]:
    for model in ["parallel","tree", "tree_joint"]:
        tasks.extend([
            ['python', '-m', 'main_MLP',
                '--pred_horizon', '30',
                '--neuro_mapping', f'train_config/MLP_{model}.nm',
                '--seed', str(seed),
                '--student_model', model
            ],
            ['python', '-m', 'main_MLP',
                '--pred_horizon', '30',
                '--seed', str(seed),
                '--student_model', model
            ],
            ['python', '-m', 'main_MLP',
                '--pred_horizon', '45',
                '--neuro_mapping', f'train_config/MLP_{model}.nm',
                '--seed', str(seed),
                '--student_model', model
            ],
            ['python', '-m', 'main_MLP',
                '--pred_horizon', '45',
                '--seed', str(seed),
                '--student_model', model
            ],
            ['python', '-m', 'main_MLP',
                '--pred_horizon', '60',
                '--neuro_mapping', f'train_config/MLP_{model}.nm',
                '--seed', str(seed),
                '--student_model', model
            ],
            ['python', '-m', 'main_MLP',
                '--pred_horizon', '60',
                '--seed', str(seed),
                '--student_model', model
            ],
            ['python', '-m', 'main_MLP',
                '--pred_horizon', '120',
                '--neuro_mapping', f'train_config/MLP_{model}.nm',
                '--seed', str(seed),
                '--student_model', model
            ],
            ['python', '-m', 'main_MLP',
                '--pred_horizon', '120',
                '--seed', str(seed),
                '--student_model', model
            ]
        ])

# Function to run a single task
def run_task(task):
    """Run a single task using subprocess.run"""
    start_time = datetime.now()
    print(f"Starting task {task} at {start_time.strftime('%d/%m/%Y, %H:%M:%S')}")
    
    result = subprocess.run(task, shell=False, capture_output=True, text=True)

    end_time = datetime.now()
    print(f"Finished task {task} at {end_time.strftime('%d/%m/%Y, %H:%M:%S')} (Duration: {end_time - start_time})")
    
    if result.returncode == 0:
        return f"Task {task} completed successfully. Output:\n{result.stdout}"
    else:
        return f"Task {task} failed with error: {result.stderr}"

if __name__ == '__main__':
    # Using ProcessPoolExecutor to run tasks concurrently
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f"{task} generated an exception: {exc}")
import time 
class logger:
    curent_level = "DEBUG"
    levels = {"DEBUG": 0, "INFO": 1,"TIME":1.5, "WARN": 2, "ERROR": 3, "NONE": 4}
    start_time = time.perf_counter()
    
    def log(level: str, msg: str):
        if level not in logger.levels:
            print(f"[{level}] {msg}")
        elif level == "TIME":
            elapsed = time.perf_counter() - logger.start_time
            print(f"[{elapsed:.2f}][{level}] {msg}")
        
        elif logger.levels[level] >= logger.levels[logger.curent_level]:
            print(f"[{level}] {msg}")
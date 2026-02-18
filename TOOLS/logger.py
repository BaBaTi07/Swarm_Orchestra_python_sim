class logger:
    curent_level = "DEBUG"
    levels = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "NONE": 4}
    
    def log(level: str, msg: str):
        if level not in logger.levels:
            print(f"[{level}] {msg}")
        
        elif logger.levels[level] >= logger.levels[logger.curent_level]:
            print(f"[{level}] {msg}")
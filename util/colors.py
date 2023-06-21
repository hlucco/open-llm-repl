colors = {
    "RED" : "\033[31m",
    "RESET" : "\033[0m",
}

def red(msg: str) -> str:
    return f"{colors['RED']}{msg}{colors['RESET']}"

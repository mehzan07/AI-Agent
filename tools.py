from agents import function_tool


@function_tool 
def calculate_total(price: float, quantity: int) -> float: 
    """Calculate the total price.""" 
    return price * quantity
from langchain_core.tools import tool



@tool
def calculate_sphere_volume(radius):
    """
    Calculate the volume of a sphere given its radius.
    
    Args:
        radius (float): The radius of the sphere
        
    Returns:
        float: The volume of the sphere
    """
    import math
    volume = (4/3) * math.pi * (radius ** 3)
    return volume

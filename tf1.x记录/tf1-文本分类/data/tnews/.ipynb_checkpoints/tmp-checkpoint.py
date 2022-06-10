import numpy as np
import io

if __name__ == "__main__":
    with io.open("toutiao_category_dev.txt", "r") as file:
        catagory = {}
        for line in file:
            data = line.strip().split("_!_")
            assert len(data) == 5
            catagory_id = data[1]
            catagory_str = data[2]
            
            if catagory_id not in catagory.keys():
                catagory[catagory_id] = catagory_str
    print(catagory)
    print(len(catagory))
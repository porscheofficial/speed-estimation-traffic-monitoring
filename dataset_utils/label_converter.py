import pickle
from pathlib import Path

import pandas as pd
import pickle_compat

pickle_compat.patch()

brno_dataset_path = "/path/to/dataset"


def list_pkl_files_in_folder(target: Path):
    pkl_list = []
    for file in target.iterdir():
        if file.is_dir():
            pkl_list += list_pkl_files_in_folder(file)
        elif file.name == "gt_data.pkl":
            pkl_list.append(file)
    return pkl_list


def map_car_to_entry(car):
    if car["valid"] is not True:
        return None
    intersection_times = [x["videoTime"] for x in car["intersections"]]
    intersection_times = sorted(intersection_times)
    if len(intersection_times) < 2:
        print(car)
    return intersection_times[0], intersection_times[1], car["carId"], car["speed"]


def main():
    pkl_to_convert = list_pkl_files_in_folder(Path(brno_dataset_path))
    for pkl in pkl_to_convert:
        out_path = Path(pkl).with_name("cars.csv")

        print(f"Converting {pkl} to {out_path}")

        with open(pkl, "rb") as f:
            pkl_obj = pickle.load(f)

        mapped_list = list(map(map_car_to_entry, pkl_obj["cars"]))
        mapped_list = filter(None, mapped_list)
        df = pd.DataFrame.from_records(mapped_list, columns=["start", "end", "carId", "speed"])
        df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

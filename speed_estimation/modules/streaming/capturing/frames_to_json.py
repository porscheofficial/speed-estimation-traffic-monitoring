import glob
import json
import os

# number extraction is hard coded. Better idea would be to delimitate order index from the filename
# by a "_"
sorted_frames = sorted(
    [os.path.basename(x) for x in glob.glob("./images/*.png")],
    key=lambda x: int(x[14:-4]),
)
json_list = {}
for i, filename in enumerate(sorted_frames[:-1]):
    json_list[str(i)] = [
        os.path.join("./images/", sorted_frames[i]),
        os.path.join("./images/", sorted_frames[i + 1]),
    ]

with open("./data/train.json", "w") as fout:
    json.dump(json_list, fout)

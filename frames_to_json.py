import json
import os
import glob
import pathlib

inputPath ='/Users/your_username/Documents/os/frames_HamptonRoads/*.png'

testImages = glob.glob(inputPath)


num = 0

for image in testImages:
    num = num + 1

print(num)

i = 0

listObj = []
filename = 'test_pytorch_HamptonRoads.json'
#dataName = 'HamptonRoads864'

#with open("train_pytorch_4KVideoOfHighwayTraffic.json", "w") as fp:


for image in sorted(testImages):
    #path = os.path.relpath(image)
    path = pathlib.Path(image).parent.resolve()
    dataName = os.path.splitext(image)[0]

    if i > num - 2:
        break

    else:
        #name1 = str(path)+'/'+ dataName +str(i)+'.png'
        #name2 = str(path)+'/'+ dataName +str(i+1)+'.png'

        name1 = dataName +str(i)+'.png'
        name2 = dataName +str(i+1)+'.png'

        new_elements = {i : [name1, name2]}

        with open(filename) as fp:
            listObj = json.load(fp)

        #print(listObj)
        
        listObj.append(new_elements)


        with open(filename, "w") as outfile:
            json.dump(listObj, outfile)    
        
        i=i+1

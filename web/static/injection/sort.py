import os
path = r"./ResNet50"
dstpath = r"./ResNet501"
for file in os.listdir(path):
    filepath = os.path.join(path,file)
    with open(filepath,"r") as fp:
        line = fp.readlines()
    newContent = []
    print(file)
    for i in range(16):
        temp = " ".join(line[0].split(" ")[(i*16):(i*16+16)])+"\n"
        newContent.append(temp)
    with open(os.path.join(dstpath,file),"w") as nfp:
        nfp.writelines(newContent)
    

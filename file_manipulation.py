f = open(".\\train_test_inputs\\nju2k_test_files_with_gt.txt","r")

mainList = f.readlines()

print(len(mainList))


"""
f = open(r".\train_test_inputs\kitti_eigen_test_files_with_gt.txt","r")

mainList = f.readlines()

f.close()

f = open(r".\train_test_inputs\kitti_eigen_test_files_with_gt.txt","w")

for line in mainList:
    if("None" in line):
        f.write(line)
    else:
        date = line[0:10]
        L = line.split(" ")
        newstr = date + "/" + L[1]
        L[1] = newstr
        finalstr = L[0] + " " + L[1] + " " + L[2]
        f.write(finalstr)

f.close()
"""

"""
f = open(r".\train_test_inputs\kitti_eigen_train_files_with_gt.txt","r")

mainList = f.readlines()

f.close()

f = open(r".\train_test_inputs\kitti_eigen_train_files_with_gt.txt","w")

for line in mainList:
    date = line[0:10]
    L = line.split(" ")
    newstr = date + "/" + L[1]
    L[1] = newstr
    finalstr = L[0] + " " + L[1] + " " + L[2]
    f.write(finalstr)

f.close()
"""
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
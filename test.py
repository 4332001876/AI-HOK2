import os
# r=os.system("sh test.sh")
r=os.popen("ls -l |wc -l","r")
# print()
# result=str(r).split(" ")
print(r.read())
# print(r)
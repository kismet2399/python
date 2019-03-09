import os
# file_path = './raw_data'
# for filename in os.listdir(file_path):
#     print(filename)
# import random
# print(random.random())
# print(random.random())
# print(random.random())
# print(random.random())
file = open('./kismet', 'r')
read = file.read()
words = read.split()
for word in words:
    print("word%s,len%i"%(word,len(word)))


import statistics
data=[]
n=int(input("Enter the number of elements: "))
for i in range(0,n):
    ele=int(input())
    data.append(ele)
data.sort()
print(data)
print(statistics.median(data))  
print(statistics.mean(data)) 

a = ndarray.arange(10.)
#print(a[1:-1])
#print(a[:-1])
a[1:-1] += a[:-2]
print(a)
#print(a[1:-1])

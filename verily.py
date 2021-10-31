def singleNumber(nums):
        return (3*sum(set([1,1,3,1,2,2,2])) - sum([1,1,3,1,2,2,2]))//2
    
singleNumber([1,1,3,1])    

def minDominoRotations(A,B):
	n = len(A)
	pos = [0] * 7  # encouters
	for i in range(n):
		pos[A[i]] += 1
		pos[B[i]] += A[i] != B[i]
	most = max(pos)
	if most != n: return -1
	target = pos.index(most)
	return min(n - A.count(target), n - B.count(target))

minDominoRotations([3,5,1,2,3], [3,6,3,3,4])

def binary_search(arr, low, high, x):
 
     if high >= low:
         mid = (high + low) // 2
         if arr[mid] == x:
            return mid
         elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
         else:
            return binary_search(arr, mid + 1, high, x)
     else:
        return -1

arr = [ 2, 3, 4, 10, 40 ]
x = 10
 
# Function call
result = binary_search(arr, 0, len(arr)-1, x)
if result != -1:
    print("Element is present at index", str(result))
else:
    print("Element is not present in array")
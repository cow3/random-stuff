import time,random,networkx,math

################
#Numbers Stuff##
################

#Given a natural number y, checks if it is prime
#Note that if y has no divisors <=sqrt(y),
#then it can't have any greater, since they
#would have to multiply by something smaller to make y.
#This function checks all numbers >=2 up to sqrt(y)
#If no divisors are found, it returns true, otherwise false.
def isPrime(y):
    if not (isinstance(y, int) or isinstance(y, long)):
        raise TypeError
    if y < 1:
        raise ValueError
        
    if y == 1:
        return False
    if y == 2:
        return True
    
    x = 1
    test = True
    while x <= y ** 0.5:
        x += 1
        if y % x == 0:
            test = False
            break
    return test

#Generates a list of the first n prime numbers, where n

def primeListUpTo(n):
    if not (isinstance(n, int) or isinstance(n, long)):
        raise TypeError
    if n < 0:
        raise ValueError
        
    if n < 3:
        return [2]
    primeList = []
    m = 1
    while True:
        m += 6
        prime = True
        prime2 = True
        for p in primeList:
            if p > m ** 0.5:
                break
            if m % p == 0:
                prime = False
                if not prime2:
                    break
            if (m - 2) % p == 0:
                prime2 = False
                if not prime:
                    break
        if prime2:
            primeList.append(m - 2)
        if prime:
            primeList.append(m)
        if primeList[-1] > n:
            primeList.pop()
            if primeList[-1] > n:
                primeList.pop()
            return [2, 3] + primeList

def primeList(n):
    if not (isinstance(n, int) or isinstance(n, long)):
        raise TypeError
    if n < 0:
        raise ValueError
        
    primeList = []
    m = 1
    leng = 0
    while leng < n - 2:
        m += 6
        prime = True
        prime2 = True
        for p in primeList:
            if p > m ** 0.5:
                break
            if m % p == 0:
                prime = False
                if not prime2:
                    break
            if (m - 2) % p == 0:
                prime2 = False
                if not prime:
                    break
        if prime2:
            primeList.append(m - 2)
        if prime:
            primeList.append(m)
        leng = len(primeList)
    if leng == n - 1:
        primeList = primeList[:-1]
    return [2, 3] + primeList

def primeSum(n):
    return sum(primeList(n))

def factorial(n):
    if not (isinstance(n, int) or isinstance(n, long)):
        raise TypeError
    if n < 0:
        raise ValueError
    prod = 1
    for i in xrange(2, n + 1):
        prod *= i
    return prod
    
def product(listOfNumbers):
    if not hasattr(listOfNumbers, '__iter__'):
        raise TypeError
    prod=1
    for n in listOfNumbers:
        if not (isinstance(n, int) or isinstance(n, float) or isinstance(n, long)):
            raise TypeError
        prod *= n
    return prod

def choose(n,r):
    if not (isinstance(n, int) or isinstance(r, int)):
        raise TypeError
    if not 0 <= r <= n:
        raise ValueError
    prod = 1
    for i in xrange(1, r + 1):
        prod *= n + 1 - i
    for i in xrange(2, r + 1):
        prod /= i
    return prod

def catalan(n):  
    # Note can ignore type and value problems, since choose will catch them.
    return choose(2 * n, n) / (n + 1)

def gcd(m,n):
    if not ((isinstance(n, int) or isinstance(n, long)) and (isinstance(m ,int) or isinstance(m, long))):
        raise TypeError
    if n < 1 or m < 1:
        raise ValueError
    if m < n:
        m, n = n, m
    while True:
        r = m % n
        if r == 0:
            return n
        else:
            m, n = n, r

def nrDivisors(n):
    return len(setDivisors(n))

def setDivisors(n):
    if not (isinstance(n, int) or isinstance(n, long)): #Ensure working with integers
        raise TypeError
    if n < 1:#Esnure integers are positive
        raise ValueError
    if n == 1:# Below method will not work if n==1, since 1%n will return 1, causing an infinite loop below. Therefore, the value for 1 has been hard-coded
        return 1
    listPrimeFactors = []
    continuing = True
    i=1
    while continuing:
       i += 1
       if n % i == 0:
           listPrimeFactors.append(i)
           n /= i# When a divisor is found, divide the current n by it as the result will have the same factors as the current n, expecting the found factor
           i -= 1
           if n == 1:
               continuing = False
    setDivisors = {1}
    for i in xrange(1, len(listPrimeFactors) + 1):
        for smallList in combinations(listPrimeFactors, i):
            setDivisors.add(product(smallList))
    return setDivisors

def primeFactors(n):
    if not (isinstance(n, int) or isinstance(n, long)):
        raise TypeError
    
    #Esnure integers are positive
    if n < 1:
        raise ValueError
    
    # Below method will not work if n==1, since 1%n will return 1, causing an infinite loop below.
    #Therefore, the value for 1 has been hard-coded
    if n == 1:
        return 1
    listPrimeFactors = []
    continuing = True
    i = 1
    while continuing:
       i += 1
       
       # When a divisor is found, divide the current n by it as the result will have
       #the same factors as the current n, expecting the found factor
       if n % i == 0:
           listPrimeFactors.append(i)
           n /= i
           i -= 1
           if n == 1:
               continuing = False
    return listPrimeFactors

def triangle(n):
    if not (isinstance(n, int) or isinstance(n, long)):
        raise TypeError
    return n * (n + 1) / 2
    
def pentagonal(n):
    if not (isinstance(n, int) or isinstance(n, long)):
        raise TypeError
    return n * (3 * n - 1) / 2    
    
def fibonacci(n):
    if not (isinstance(n, int) or isinstance(n, long)):
        raise TypeError
    if n == 1:
        return 1
    if n == 0:
        return 0
    return fibonacci(n - 1) + fibonacci(n - 2)
    
def lucas(n):
    if not (isinstance(n, int) or isinstance(n, long)):
        raise TypeError
    if n == 1:
        return 1
    if n == 0:
        return 2
    return lucas(n - 1) + lucas(n - 2)

def fibonacciWithGivenStart(a, b, n):
    if not (isinstance(n, int) or isinstance(n, long)) and \
        (isinstance(a, int) or isinstance(a, long) or \
        isinstance(a, float)) and  (isinstance(b, int) or \
        isinstance(b, long) or isinstance(b, float)):
        raise TypeError
    if n < 0:
        return ValueError
    if n == 1:
        return b
    if n == 0:
        return a
    return fibonacciWithGivenStart(a, b, n - 1) + fibonacciWithGivenStart(a, b, n - 2)
    
def goldbachAny(even):
    if not(isinstance(even, int) or (isinstance(even, long))):
        raise TypeError
    if even < 3 or even % 2 == 1:
        raise ValueError
    for x in range(1, even / 2):
        if isPrime(x) and isPrime(even - x):
            return[x, even - x]
    return "you've disproved the goldbach conjecture with " + str(even) + "!!!!!!!!!!"

def goldbachAll(even):
    if not(isinstance(even, int) or (isinstance(even, long))):
        raise TypeError
    if even < 3 or even % 2 == 1:
        raise ValueError
    answers = []
    for x in range(1, even / 2):
        if isPrime(x) and isPrime(even - x):
            answers.append([x ,even - x])
    if answers == []:
        return "you've disproved the goldbach conjecture with " + str(even) + "!!"
    else:
        return answers
    
def isTwinPrime(y):
    if not(isinstance(y, int) or (isinstance(y, long))):
        raise TypeError
    if y < 1:
        raise ValueError
    return isPrime(y) and (isPrime(y + 2) or isPrime(y - 2))
    
def isCousinPrime(y):
    if not (isinstance(y, int) or (isinstance(y, long))):
        raise TypeError
    if y < 1:
        raise ValueError
    return isPrime(y) and (isPrime(y + 4) or isPrime(y - 4)) 
    
def isSexyPrime(y):
    if not (isinstance(y, int) or (isinstance(y, long))):
        raise TypeError
    if y < 1:
        raise ValueError
    return isPrime(y) and (isPrime(y + 6) or isPrime(y - 6))

def isnDistancePrime(n, p):
    if not (isinstance(p, int) or (isinstance(p, long))):
        raise TypeError
    if p < 1:
        raise ValueError
    return isPrime(p) and (isPrime(p + n) or isPrime(p - n))
    
def derangements(n):
    if not (isinstance(n, int) or (isinstance(n, long))):
        raise TypeError
        
    #Note factorial checks n is an int or long  and that n>=0
    return sum([factorial(n) * (-1) ** i / factorial(i) for i in xrange(0, n + 1)])
           
def polygonal(sides, n):
    if not ((isinstance(n, int) or isinstance(n, long)) and \
           (isinstance(sides, int) or isinstance(sides, long))):
        raise TypeError
    if n < 0 or sides < 1:
        raise ValueError
    return (n ** 2 * (sides - 2) - n * (sides - 4)) / 2

def carmichaelCheck(n):
    if isPrime(n):
        return False
    for x in range(n):
        if (x ** n) % n != x:
            return False
    return True

def carmichael(n):
    count = 0
    x = 0
    while count < n:
        x += 1
        count += carmichaelCheck(x)
    return x

def isSquare(n):
    if not(isinstance(n, int) or (isinstance(n, long))):
        raise TypeError
    return n == round(n ** (0.5)) ** 2
       
def isCube(n):
    if not(isinstance(n, int) or (isinstance(n, long))):
        raise TypeError
    return n == round(n ** (1. / 3)) ** 3
        
def isnthPower(m,n):
    if not(isinstance(n, int) or (isinstance(n,long))):
        raise TypeError
    return m == round(m ** (1. / n)) ** n

def Recaman(n):
    if not(isinstance(n, int) or (isinstance(n, long))):
        raise TypeError
    if n == 0:
        return 0
    if n < 1:
        raise ValueError
    sequence = [0]
    while len(sequence) < n + 1:
        case1 = sequence[-1] - len(sequence)
        case2 = sequence[-1] + len(sequence)
        if case1 > 0 and not case1 in sequence:
            sequence.append(case1)
        else:
            sequence.append(case2)
    return sequence[-1]
    
def isHappy(n):
    if not(isinstance(n, int) or (isinstance(n, long))):
        raise TypeError
    if n < 1:
        raise ValueError
    if n == 1:
        return True
    sequence = [n]
    while True:
        sequence.append(sum(map(lambda x : int(x) ** 2, str(sequence[-1]))))
        if sequence[-1] == 1:
            return True
        if sequence[-1] in sequence[:-2]:
            return False

def happy(n):
    if not(isinstance(n, int) or (isinstance(n, long))):
        raise TypeError
    if n < 1:
        raise ValueError
    count = 0
    x = 0
    while count < n:
        x += 1
        if isHappy(x):
            count += 1
    return x

def lucasLehmer(p):
    if not(isinstance(p, int) or (isinstance(p, long))):
        raise TypeError
    if p < 0:
        raise ValueError
    if not isPrime(p):
        return False
    if p == 2:
        return True
    a = 4
    b = 2 ** p - 1
    for x in range(p - 2):
        a=((a ** 2) - 2) % b
    return a == 0
    
def mersennePrime(n):
    if not (isinstance(n, int) or isinstance(n, long)):
        raise TypeError
    if n < 1:
        raise ValueError
    check = 0
    count = 0
    while count < n:
        check += 1
        if lucasLehmer(check):
            count += 1
    return 2 ** check - 1
        
def perfect(n): 
    prime = mersennePrime(n)
    return prime * (prime + 1) / 2

######################
#Shuffling Algorithms#
######################

#Returns a 'random' permutation of a list
#without altering the list.
#It starts by defining a list of the appropriate size
#then setting the ith element of theat list to a random
#element of the copy of the input list.
#It then deletes the entry of the copy of the input
#list to ensure the smae element is not repeated
def shuffle1(List):
    if not isinstance(List, list):
        raise TypeError

    copiedList = List[:]
    #Copy list
    
    outputList=range(len(List))
    for i in xrange(len(List)):
        e = random.choice(copiedList)
        outputList[i] = e
        copiedList.remove(e)
    return outputList

def SwapShuffle(L):
    n = len(L)
    currentList = L[:] 
    #Copies the input list into a new variable so that it is not altered
    #This avoids needing to import copy
    
    for x in xrange(n ** 2):
        #This repeats n^2 times due to for loop
        #Given i as any integer <n, j, a random integer in the same range will equal with probability 1/n
        #so the two elements of the list that will be swapped will be the
        #same element with probability 1/n        
        j = random.randrange(n)
        i = random.randrange(n)
        
        #Use k to store a variable in order to swap the ith and
        #jth elements of L
        k = currentList[i]
        currentList[i] = currentList[j]
        currentList[j] = k
    return currentList





#####################
#Sorting Algorithms##
#####################

#Checks if list is sorted.
def isSorted(List):
    if not isinstance(List, list):
        raise TypeError

    return List == sorted(List)

#Performs a bogosort.
#This involves repeatedly shuffling a list until the list is sorted
#Note that isSorted function checks that List is a list
#Continually shuffles the list until it is sorted
def bogoSort(List):
    x = isSorted(List)
    while x == False:
        random.shuffle(List)
        x = isSorted(List)
    return List
        
#Performs a radix sort.
#Take the least significant digit of each element.
#organize the elements based on that digit, but otherwise keep 
#the original order. Repeat the organizing process with 
#each more significant digit.
def radixSort(List):
    if not isinstance(List, list):
        raise TypeError
    for i in List :
        if not isinstance(i, int) or isinstance(i, int):
            raise TypeError
    
    output = List[:]
    #Copy List.

    output = [list(reversed([int(i) for i in list(str(j))])) for j in output]
    #Converts r in to a list of reversed lists of digits (for easier sorting).
    longest = max([len(i) for i in output])
    #Finds the longest list in the list 
    #(equivalently, the number with the most digits).

    for i in output:
        while len(i) < longest:
            i.append(0)
    #Makes all list in the lists of equal length, 
    #without changing the value they represent.

    for i in xrange(longest):
        x = 1
        while x:
            x = 0
            for j in range(len(output) - 1):
                if output[j][i] > output[j + 1][i]:
                    x += 1
                    output[j], output[j + 1] = output[j + 1], output[j]
                    #Cycles through the digits and sorts accordingly.

    for i in range(longest):
        for j in output:
            j[i] = j[i] * (10 ** i)
    for j in range(len(output)):
        output[j] = sum(output[j])
        #Converts the lists back into numbers
    return output

#Perform a Quicksort.
#Any list of length 0 or 1 is already sorted
#If List is not length 0 or 1, call the first entry of the list the pivot
#Compare all entries of List to the pivot
#If they are smaller they go into the list called 'small'
#Otherwise they go into the list called 'big'
#quickSort is then recursively called on small and big
#The output is a list, first containing sorted small then pivot then sorted big
#Note that there is no need to copy List, as it is not altered
def quickSort(List):
    if not isinstance(List, list):
        raise TypeError
    if len(List) < 2:
        return List

    pivot = List[0]
    small = []
    big = []
    for i in xrange(1, len(List)):
        if List[i] < pivot:
            small.append(List[i])
        else:
            big.append(List[i])
    sortedSmall = quickSort(small)
    sortedBig = quickSort(big)
    sortedSmall.append(pivot)
    return sortedSmall + sortedBig
        
#Merge 2 sorted lists.
def merge(f,s):
    r = []
    while len(f) > 0 and len(s) > 0:
        if f[0] < s[0]:
            r.append(f[0])
            f.pop(0)
        else:
            r.append(s[0])
            s.pop(0)
        #Adds the smallest of the first elements in the 2 lists, to the new list.
    if len(f) != 0:
        r.extend(f)
    elif len(s) != 0:
        r.extend(s)
    #Adds the remainder of one list once the other is empty.
    return r

#Perform a mergesort.
def mergeSort(L):
    a = len(L)
    if a < 2:
        return L
        #Returns L if we know it's sorted due to it's size.
    if a > 2:
        F = L[:(a / 2)]
        S = L[(a / 2):]
        #Splits into two lists of roughly equal size.
        f = mergeSort(F)
        s = mergeSort(S)
        #Performs Mergesort on these 2 lists.
    else:
        if L[0] > L[1]:
            L[0], L[1] = L[1], L[0]
            return L
        else:
            return L
        #Sorts a 2 element list.
    r = merge(f, s)
    #Merge the two sorted lists.
    return r
        
        
        
#Heaping.
def heap(k, L):
    if len(L) > (2 * k) + 2:
        #Checks if the kth element has 2 children
        M = max(L[(2 * k) + 1], L[(2 * k) + 2])
        if L[k] < M:
            m = L.index(M)
            L[k], L[m] = L[m], L[k]
            heap(m, L)
            #Swaps kth element with it's largest child if larger than itself

    elif len(L) == (2 * k) + 2:
        #Checks if the kth element has only 1 child.
        M = L[(2 * k) + 1]
        if L[k] < M:
            m = L.index(M)
            L[k], L[m] = L[m], L[k]
            heap(m, L)
            #Swaps the kth element with it's child if it's child is larger.
    return L

#Perform heapsort.
def heapSort(L):
    H = L[:]
    #Copy list

    for i in xrange(len(H) - 1, -1, -1):
        H = heap(i, H)
        #Performs Heap on the element from last to first, converting H into a Heap.
    K = []
    #Create new list.
    while len(H) > 0:
        K = [H[0]] + K
        #Add the first(and largest) element in H to K. 
        H[0] = H[-1]
        #Replace the first element of H with the last
        H.pop(-1)
        #Delete the last element.
        if len(H) > 0:
            H = heap(0, H)
            #Convert H back to a Heap.
    return K

#Performs a bubble sort.
def BubbleSort(n):
    r = n[:]
    #Copy list
    x = 1
    #x is using for checking how many changes have been made per pass.
    while x:
        #Loops until no changes are made, which is when the list is sorted.
        x = 0
        #resets x.
        for i in xrange(len(n) - 1):
            if r[i] > r[i + 1]:
                #loops over list and compares consecutive elements.
                x += 1
                r[i], r[i + 1] = r[i + 1], r[i]
                #if the second element is smaller, Swaps them and increments x.
    return r

#################
##Linear Maths###
#################

def isMatrix(listOfRows):
    if not isinstance(listOfRows, list):
        raise TypeError
    check = len(listOfRows[0])
    for x in listOfRows:
        if not isinstance(x, list):
            raise TypeError
        if not len(x) == check:
            return False
    for x in listOfRows:
        for y in x:
            if not isinstance(y,int) or isinstance(y,long) or isinstance(y,float):
                raise False
    return True

def stringReprMatrix(matrix):
    if not isMatrix(matrix):
        raise TypeError
    output='['
    for row in matrix:
        for entry in row:
            output += '%s ' % entry
        output = output[:-1]
        output += ']\n['
    return output[:-1]

def copyMatrix(matrix):
    if not isMatrix(matrix):
        raise TypeError
    output = []
    for row in matrix:
        output.append(row[:])
    return output

def dimesionMatrix(matrix):
    if not isMatrix(matrix):
        raise TypeError
    return (len(matrix),len(matrix[0]))        

#adds two given matrices together(checking if the entries are matrices)
#and checking if the dimensions are appropriate for addition
#returns the sum
def matrixAdd(matrix1, matrix2):
    m, n = dimesionMatrix(matrix1), dimesionMatrix(matrix2)
    if not m==n:
        raise TypeError
    ans = copyMatrix(matrix1)
    for x in xrange(m[0]):
        for y in xrange(m[1]):
            ans[x][y]=matrix1[x][y]+matrix2[x][y]
    return ans

#Takes as input two matrices (as lists of lists)
#Checks if they are matrices, and if they can be multiplied
#If so multiplies them
#It does this by first taking the set range(m1), where
#m1 is the number of rows of M and mapping each element in
#it to M_(xz)*M_(zy), for all x up to m1, y up to n2.
#All of these results are then stored in a list tempAnswer
#The list range(n2) is then mapped using the map that takes
#in x and sends it the the partition of tempAnswer
#from x*m1 to (x+1)*m
def matrixMultiply(M, N):
     #dimension checks matrix is matrix
     m, n = dimesionMatrix(M), dimesionMatrix(N)
     if not m[1] == n[0]:
         raise TypeError

     #stores answer as a list which will be converted into a matrix  
     tempAnswer=[]   
     
     for x in xrange(m[0]):
         for y in xrange(n[1]):
    
     #loops through colums of M and rows of N adding products
             tempAnswer.append(sum(map(lambda z : M[x][z] * N[z][y],
                                       range(m[0]))))
     return map(lambda x:tempAnswer[x * m[0] : (x + 1) * m[0]], range(n[1]))

def isSquareMatrix(matrix):
    #dimension checks matrix is matrix
    d = dimesionMatrix(matrix)
    return d[0] == d[1]

#Finds the determinant of a matrix using the Leibniz formula
def determinant(matrix):
    if not isSquareMatrix(matrix):
        raise TypeError
    n = len(matrix)
    output = 0
    if n == 1:
        return matrix[0][0]
    for sigma in symmetricGroupList(n):
        if isEven(sigma):
            sign = 1
        else:
            sign = -1
        prod = 1
        for i in xrange(n):
            j = sigma[i] - 1
            prod *= matrix[i][j]
        output += sign * prod
    return output

def isInvertible(matrix):
    #Note that determinant will catch non-square matrices
    if determinant(matrix)==0:
        return False
    return True

def transposeMatrix(matrix):
    if not isSquareMatrix(matrix):
        raise TypeError
    n = len(matrix)
    output = copyMatrix(matrix)
    for i in xrange(n):
        for j in xrange(n):
            output[i][j] = matrix[j][i]
    return output

def adjugate(matrix):
    if not isInvertible(matrix):
        raise TypeError
    output = copyMatrix(matrix)
    n = len(matrix)
    for i in xrange(n):
        for j in xrange(n):
            tempMatrix = copyMatrix(matrix)
            for k in xrange(n):
                del tempMatrix[k][j]            
            del tempMatrix[i]
            output[i][j] = determinant(tempMatrix)
    return output

def inverseMatrix(matrix):
    if not isInvertible(matrix):
        raise TypeError
    adj = adjugate(matrix)
    det = determinant(matrix)
    if isinstance(det, int):
        integer = True
    for row in adj:
        for entry in row:
            if isinstance(entry, int) and integer:
                if entry % det == 0:
                    entry /= det
                else:
                     entry /= float(det)
            else:
                entry /= float(det)
    return adj
    
def isVector(V):
    if not isinstance(V, list):
        return False
    for x in V:
        if not isinstance(x, int) or isinstance(x, long) or isinstance(x, float):
            raise TypeError
    return True
    
def vectorDotProduct(V, U):
    if not isVector(V) and isVector(V):
        raise TypeError
    if not len(V) == len(U):
        raise TypeError
    return sum(map(lambda x : V[x] * U[x], range(len(U))))

def vectorCrossProduct(V,U):
    if not isVector(V) and isVector(V):
        raise TypeError
    if not len(V) == len(U) == 3:
        raise TypeError
    return [V[1] * U[2] - U[1] * V[2],
            U[0] * V[2] - V[0] * U[2],
            V[0] * U[1] - U[0] * V[1]]

def convertToColumnMatrix(V):
    if not isVector(V):
        raise TypeError
    return map(lambda x:[x], V)

def convertToVector(M):
    if not isMatrix(M):
        raise TypeError
    for x in M:
        if not len(x) == 1:
            raise TypeError
    return map(lambda x : x[0], M)
    
############
#Graphs#####
############

#Creates the empty graph on n vertices
#Does this by adding vertices for i
#from 0 to n-1    
def emptyGraph(n):
    if not isinstance(n, int):
        raise TypeError
    if n < 0:
        raise ValueError
    graph = networkx.Graph()
    for i in xrange(n):
        graph.add_node(i)
    return graph

#Takes a graph, and returns the values of the
#degree dictionary
def degreeSequence(graph):
    if not isinstance(graph, networkx.classes.graph.Graph):
        raise TypeError
    return graph.degree().values()

#Finds all vertices of a graph with degree d.
#Takes a graph and a non-negative integer d, and searches the
#dictionary for keys with value d.
def degd(graph, d):
    if not isinstance(graph, networkx.classes.graph.Graph) and \
    isinstance(d, int):
        raise TypeError
    if d < 0:
        raise ValueError
    output = []
    for key in graph.degree().keys():
        if graph.degree()[key] == d:
            output.append(key)
    return output


#Finds the smallest degree of a graph
#Calls degd on graph using i as the non-negative integer
#Each time the result is [], i is incremeneted by 1
#When a non-empty list is found, the result is returned.
def minimalDegree(graph):
    #Note that type of graph will be checked by degd
    i = -1
    listOfMinimalVertices = []
    while listOfMinimalVertices == []:
        i += 1
        listOfMinimalVertices = degd(graph, i)
    return i, listOfMinimalVertices
        
###################
#Lists Stuff#######
###################
        
#Taken from itertools source code
def combinations(iterable,r):
    if not hasattr(iterable, '__iter__'):
        raise TypeError
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)

def permutations(iterable):
    if not hasattr(iterable, '__iter__'):
        raise TypeError
    List = list(iterable)
    l = len(List)
    P = []
    for i in List:
        for j in range(factorial(l - 1)):
            P.append([i])
    done = False
    while not done:
        i = 0
        for k in range(len(P)):
            C = [j for j in List if j not in P[k]]
            P[k].append(C[i])
            if i != len(C) - 1:
                i += 1
            else:
                i = 0
        done = True
        for p in P:
            if len(p) != l:
                done = False
    return P

#Taken from itertools source code
#==============================================================================
# def permutations(iterable, r=None):
#     # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
#     # permutations(range(3)) --> 012 021 102 120 201 210
#     pool = tuple(iterable)
#     n = len(pool)
#     r = n if r is None else r
#     if r > n:
#         return
#     indices = range(n)
#     cycles = range(n, n-r, -1)
#     yield tuple(pool[i] for i in indices[:r])
#     while n:
#         for i in reversed(range(r)):
#             cycles[i] -= 1
#             if cycles[i] == 0:
#                 indices[i:] = indices[i+1:] + indices[i:i+1]
#                 cycles[i] = n - i
#             else:
#                 j = cycles[i]
#                 indices[i], indices[-j] = indices[-j], indices[i]
#                 yield tuple(pool[i] for i in indices[:r])
#                 break
#         else:
#             return
# 
#==============================================================================

def permutationsOfSublist(iterable,r):
    if not hasattr(iterable, '__iter__'):
        raise TypeError
    List = list(iterable)
    l = len(List)
    assert r <= l
    P = []
    for i in iterable:
        for j in xrange(factorial(l - 1)):
            P.append([i])
    done = False
    while not done:
        i = 0
        for k in xrange(len(P)):
            C = [j for j in List if (j not in P[k])]
            P[k].append(C[i])
            if i != len(C) - 1:
                i += 1
            else:
                i = 0
        done = True
        for p in P:
            if len(p) != r:
                done = False
    for k in xrange(len(P)):
        a = [str(j) for j in P[k]]
        P[k] = ''.join(a)
    P = set(P)
    P = list(P)
    for k in xrange(len(P)):
        P[k] = [int(j) for j in P[k]]
    return P

def isPermutation(List):
    if not isinstance(List, list):
        return False
    n = len(List)
    return set([x + 1 for x in xrange(n)]) == set(List)

def disjointCycle(permutation):
    if not isPermutation(permutation):
        raise TypeError
    identityPermutation = map(lambda x : x + 1, range(len(permutation)))
    if permutation == identityPermutation:
        return set()
    currentCycle = [1, permutation[0]]
    doneList = set()
    initialOutput = []
    while len(doneList) != len(permutation):
        doneList.add(currentCycle[0])
        doneList.add(currentCycle[1])
        if currentCycle[-1] == currentCycle[0]:
            for i in xrange(len(permutation)):
                k = -1
                if (i + 1) not in doneList:
                    k = i
                if k == -1:
                    break
                currentCycle = [k, permutation[k - 1]]
                continue
        while currentCycle[-1] != currentCycle[0]:
            currentCycle.append(permutation[currentCycle[-1] - 1])
        if len(currentCycle) >= 3:
            initialOutput.append(currentCycle[:-1])
        for i in permutation:
            if i not in doneList:
                currentCycle=[i, permutation[i - 1]]
    output = []
    sortedOutput = []
    for cycle in initialOutput:
        if not tuple(sorted(cycle)) in sortedOutput:
            sortedOutput.append(tuple(sorted(cycle)))
            output.append(tuple(cycle))
    return set(output)

def isEven(permutation):
    #disjointCycle will catch non-permutations
    n = len(permutation)
    if permutation == [x + 1 for x in xrange(n)]:
        return True
    disjointCyles = disjointCycle(permutation)
    n = 0
    for cycle in disjointCyles:
        if len(cycle) % 2 == 0:
            n += 1
    return n % 2 == 0

def symmetricGroupList(n):
    if not isinstance(n, int):
        raise TypeError
    if n < 1:
        raise ValueError
    return permutations([x + 1 for x in xrange(n)])

################
##Geometry######
################

#Note this does not count lines as triangles.
#If any two sides' lengths sum to a number greater than the
#length of the third, then it is possible to make a triangle
#with these side lengths.
def isEuclideanTriangle(side1, side2, side3):
    for side in [side1, side2, side3]:
        if not isinstance(side, int) or isinstance(side,long) \
        or isinstance(side,float):
            raise TypeError
        listOfSides = [side1, side2, side3]
        listOfSides.remove(side)
        if sum(listOfSides) <= side:
            return False
    return True

#Use Heron's Formula
#First check if the triangle exists
def areaOfEuclideanTriangle(side1, side2, side3):    
    if not isEuclideanTriangle([side1, side2, side3]):
        raise ValueError
    s = (side1 + side2 + side3) / 2.0
    return (s * (s - side1) * (s-side2) * (s - side3)) ** 0.5

#Uses the fact that hyperbolic triangles vertices' add up to <pi
def isHyperbolicTriangle(angle1, angle2, angle3):
    angles = [angle1, angle2, angle3]
    for angle in angles:
        if not (isinstance(angle, float) or isinstance(angle, int) or \
               isinstance(angle, long)):
            raise TypeError
    return sum(angles) < math.pi

#Uses Gauss-Bonnet Theorem
def areaOfHyperbolicTriangle(angle1, angle2, angle3):
    if not isHyperbolicTriangle(angle1, angle2, angle3):
        raise ValueError
    return math.pi - angle1 - angle2 - angle3

#################
##Analysis#######
#################        
def guessLimit(sequence, epsilon, breakLimit = 300000000000000):
    
    #Ensure input is function
    if not (hasattr(sequence, '__call__') or isinstance(epsilon, float)):
        raise TypeError
    n = 0
    while n < breakLimit:
       n += 1
       if abs(sequence(n) - sequence(n + 1)) < epsilon: 
           if abs(sequence(n) - sequence(n + 1)) < epsilon and \
              abs(sequence(n) - sequence(n + 2)) < epsilon and \
              abs(sequence(n) - sequence(n + 3)) < epsilon and \
              abs(sequence(n + 1) - sequence(n + 3)) < epsilon and \
              abs(sequence(n + 1) - sequence(n + 4)) < epsilon:
               return sequence(n)     
        
####################
##Random Crap#######
####################
        
def createDeckOfPlayingCards():
    deck = []
    for i in range(2, 11) + ['jack', 'queen', 'king', 'ace']:
        for j in ['spades', 'hearts', 'diamonds', 'clubs']:
            deck.append((i, j))
    return deck

def benchmark(func, args, trials = 1):

#Returns a list of floats (of length trials) indicating the time taken in
#milliseconds to run func(args).
#The arguments will be given as a list, in case the function takes
# in more than one argument
    
    if not isinstance(args, list) and isinstance(trials, int) and \
           hasattr(func, '__call__'):
        raise TypeError
    if trials < 1:
        raise ValueError
    
    result = []
    # The list of times
    for n in xrange(trials):
        start = time.time()
    # Measure the time taken to
        func(*args)
    # by func(arg)
        end = time.time()
        result.append((end - start) * 1000)
    # Convert to milliseconds
    return result

############################
##Groups and Permutations###
############################
class Perm:
    # private data for degree and largest moved point
    _deg = None
    _lmp = None

    def __init__(self, *args):
        if len(args) == 0:
            self._deg = 0
            self.image = []
        elif type(args[0]) is tuple:
            #check args and find degree
            deg = 0
            for tup in args:
                if not type(tup) is tuple:
                    raise ValueError
                if len(tup) > 0 and max(tup) > deg:
                    deg = max(tup)
            # check injective
            seen = [False] * deg

            for tup in args:
                for i in tup:
                    if seen[i - 1]:
                        raise ValueError
                    else:
                        seen[i - 1] = True
            # install the values
            self.image = range(deg)
            for tup in args:
                for i in xrange(len(tup)):
                   self.image[tup[i] - 1] = tup[(i + 1) % len(tup)] - 1
            self._deg = deg
        elif type(args[0]) is int:
            # permutation defined by list of images
            self.image = list(args)
        else:
            raise ValueError
    def __pow__(self, i):
        '''power a Perm by an integer '''
        if not type(i) is int:
            raise TypeError
        if i == 0:
            return Perm()
        if i > 0:
            perm = self.copy()
            j = 0
            while j < i - 1:
                j += 1
                perm = perm * self
            return perm
        elif i < 0:
            return self.inverse() ** -i

    def __eq__(self, right):
        '''check equality of Perms'''
        deg = max(self.degree(), right.degree())
        for i in xrange(deg):
            if self[i] != right[i]:
                return False
        return True

    def __ne__(self, right):
        '''check inequality of Perms'''
        deg = max(self.degree(), right.degree())
        deg = max(self.degree(), right.degree())
        for i in xrange(deg):
            if self[i] != right[i]:
                return True
        return False

    def __getitem__(self, index):
        '''find the image of <index> under <self>, INTERNAL ONLY.
           returns self.image[index] which is shifted by 1!
        '''
        if index < self.degree():
            return self.image[index]
        else:
            return index

    def __repr__(self):
        ''' print out as a product of disjoint cycles'''
        out = ''
        seen = [False] * self.degree()
        for i in xrange(self.degree()):
            if not seen[self[i]]:
                seen[i] = True
                j = self[i]
                if j <> i:
                    out += '(' + str(i + 1)
                    while self[j] <> i:
                        seen[j] = True
                        out += ' ' + str(j + 1)
                        j = self[j]
                    seen[j] = True
                    out += ' ' + str(j+1) + ')'
        if out == '':
            out = '()'
        return out

    def __mul__(self, right):
        deg = max(self.degree(), right.degree())
        image = range(deg)
        for i in xrange(deg):
            image[i] = self[right[i]]
        return Perm(*image)

    def __hash__(self):
        return hash(tuple(self.image[0:self.lmp()]))

    def copy(self):
        '''copy a Perm, INTERNAL ONLY'''
        return Perm(*self.image)

    def inverse(self):
        '''invert a Perm, INTERNAL ONLY, use ** -1 externally'''
        image = range(self.degree())
        for i in xrange(self.degree()):
            image[self[i]] = i
        return Perm(*image)

    def lmp(self):
        '''find the largest moved point of the perm, INTERNAL ONLY
        '''
        if self._lmp == None:
            for i in xrange(self.degree(), 0, -1):
                if self[i] != i:
                    self._lmp = i
        return self._lmp

    def degree(self):
        '''find the degree of a perm, i.e. the largest value in the list of
        images. This is not necessarily the same as the largest moved point
        (i.e. it can be larger)
        '''
        if self._deg == None:
            deg = 0
            for i in self.image:
                if i + 1 > deg:
                    deg = i + 1
            self._deg = deg
        return self._deg

    def hit(self, i):
        if type(i) is int:
            if  i >= 0:
                return self[i - 1] + 1
            else:
                raise ValueError
        else:
            raise TypeError
            
        

 ################################################################################
 # Symmetric groups
 ################################################################################

class SymmetricGroup:
    _nr_next_emitted = 0
    _size = None
    def __init__(self, deg):
        self._deg = deg
        self._current = range(self.degree() - 1, 0, -1)
        self._transpositions = []
        i = 0
        while i < self.degree():
            self._transpositions.append([None] * self.degree())
            i += 1
    def identity(self):
        '''return the identity of the group'''
        return Perm()
    def __repr__(self):
        return "<symmetric group on " + str(self.degree()) + " points>"
    def __contains__(self, perm):
        if isinstance(perm, Perm):
            return perm.degree() <= self.degree()
    def degree(self):
        '''returns the number of points acted on'''
        return self._deg
    def size(self):
        '''returns the size of the group'''
        if self._size == None:
            self._size = factorial(self.degree())
        return self._size
    def transpose(self, i, j):
        '''This is a cache for transpositions to speed things up.
           Returns the transposition (i, j) when i < j'''
        if i > j:
            i, j = j, i
        if not isinstance(self._transpositions[i][j], Perm):
            if i != j:
                self._transpositions[i][j] = Perm((i + 1, j + 1))
            else:
                self._transpositions[i][j] = Perm()
        return self._transpositions[i][j]

    def __iter__(self):
        self._current = range(self.degree() - 1, 0, -1)
        return self
    def next(self):
        '''returns the next element of the group. Uses basic stabiliser chain,
           and transpositions.
        '''
        if self._nr_next_emitted == self.size():
            self._nr_next_emitted = 0
            raise StopIteration
        self._nr_next_emitted += 1
        pos = self.degree() - 1
        perm = Perm()
        while pos > 0:
            pos -= 1
            self._current[pos] = (self._current[pos] + 1) % (self.degree() - pos)
            perm *= self.transpose(pos, self._current[pos] + pos)
            if self._current[pos] != 0:
                break
        while pos > 0:
            pos -= 1
            perm *= self.transpose(pos, self._current[pos] + pos)
        return perm

 ################################################################################
 # IsSymmetricGroup
 ################################################################################

def IsSymmetricGroup(obj):
    '''check the object is a group (within the narrow definition of the things
       implemented in this file).
    '''
    return isinstance(obj, SymmetricGroup)

def IsPerm(obj):
    '''check the object is a group (within the narrow definition of the things
       implemented in this file).
    '''
    return isinstance(obj, Perm)

 ################################################################################
 # Subsets
 ################################################################################

def subsets(group, size):
    '''returns the subsets of the group <group> of size <size>, as a list of
       lists.
    '''
    return list(map(list, combinations(group, size)))

 #########################
 #Cayley Tables
 #####################
def isCayleyTable(table):
  if not type(table) is list : 
      return False
  else:
      check=True
      for i in xrange(0,len(table)):
          if not len(table[i]) == len(table):
              check = False
          for j in xrange(0,len(table[i])) :
              if type(table[i][j]) is int:
                  if table[i][j] > len(table[i])-1:
                      check = False
              else:
                  check=False
      return check

#returns a random Cayley table over the set {0,1.....,n-1} for any n
def randomCayleyTable(n):
    if not isinstance(n,int):
        raise TypeError
    if n<1:
        raise ValueError
    table = []
    for i in xrange(n):
        table.append([])
        for j in xrange(n):
            table[i].append(random.randrange(n-1))              
    return table 

def isAssociativeCayleyTable(table):
    if not isCayleyTable(table):
        raise TypeError
    else:
        for i in xrange(len(table)):
            for j in xrange(len(table)):
                for k in xrange(len(table)):
                    if not table[table[i][j]][k] == table[i][table[j][k]]:
                        return False 
        return True

def nonAssociativeTriples(table):
    if not isCayleyTable(table):
        raise TypeError
    output=[]
    for i in xrange(len(table)):
        for j in xrange(len(table)):
            for k in xrange(len(table)):
                if not table[table[i][j]][k] == table[i][table[j][k]]:
                    output.append((i,j,k))
    return output

def identityCayleyTable(table):
    if not isCayleyTable(table):
        raise TypeError
    n=len(table)
    for i in xrange(n):
        if table[i] ==xrange(n):
            d2 = []
            for j in xrange(n):
                d2.append(table[j][i])
            if d2 == xrange(n):
                return i
    return -1

def isLeftCancellativeCayleyTable(table):
    if not isCayleyTable(table):
        return False
    n =len(table)
    for i in range(n):
        seen = [False] * n
        for j in range(n):
            if not seen[table[i][j]]:
                seen[table[i][j]] =True
            else:
                return False
    return True
# function above checks if an input is left cancellative and returns true/false accordingly
    
def isRightCancellativeCayleyTable(table):
    if not isCayleyTable(table):
        return False
    n =len(table)
    for i in range(n):
        seen = [False] * n
        for j in range(n):
            if not seen[table[j][i]]:
                seen[table[j][i]]=True
            else:
                return False
    return True
# function above checks if an input is right cancellative and returns true/false accordingly
    
def isCancellativeCayleyTable(table):
    if isLeftCancellativeCayleyTable(table) == True and isRightCancellativeCayleyTable(table) == True:
       return True
    return False      
# function above checks if an input is cancellative and returns true/false accordingly

def inverseCheck(table):
    check = False
    if not isCayleyTable(table):
        return False
    if isCayleyTable(table) == -1:
        return False
    for i in xrange(len(table)):
        for j in xrange(len(table)):
            if table[i][j] == identityCayleyTable(table) and table[j][i]== identityCayleyTable(table):
                check = True
        if check == False :
            return False
        check = False
    return True
# function above checks if all the elements of the input have inverses and returns true/false accordingly

def isGroup(table):
        if isAssociativeCayleyTable(table) and inverseCheck(table):
            return True
        return False
# function above checks if the input is a group and returns true/false accordingly

def isAbelian(table):
    if not isCayleyTable(table):
        raise TypeError
    for i in range(len(table)):
        for j in range(len(table)):
            if not table[i][j] == table[j][i]:
                return False
    return True      
# function above checks if the input is commutative true/false accordingly

def orderElementCayleyTable(table , i): 
    if not isGroup(table) or not type(i) is int:
        raise TypeError
    if not(0 <= i and i <len(table)):
        raise ValueError
    j=i
    order = 1
    for l in range(len(table)) :
         order = order + 1
         j = table[j][i]
         if j == identityCayleyTable(table):
             return order
    return -1
# function above finds the order of a given variable i in a Cayley table of a group called table.

def orderGroupElement(G,x):
    if not (IsSymmetricGroup(G) and x in G):
        raise ValueError
    order = 1
    while 1==1:
        if x**order == G.identity():
             return order
        order = order + 1
# function above finds the order of a given element x from any of the groups from the groups module
        
def isSubgroup(G,H):
    if not (IsSymmetricGroup(G) and type(H) is list):
        raise ValueError
    for h in H:
        if not h in G :
            return False
        if not h ** -1 in H:
            return False
        for g in H:
            if not h * g in H:
                return False
    return True

def isCyclicSubgroup(G, H):
    if not isSubgroup(G, H):
        return False
    for h in H:
        if orderGroupElement(G,h) == len(H):
            return True
    return False

def isAbelianSubgroup(G,H):
    if not isSubgroup(G,H):
        return False
    K = [x for x in H]
    for i in K:
        for j in K:
            if not i*j == j*i:
                return False
    return True
        

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Say "Hello, World!" With Python
print("Hello, World!")


# In[ ]:


#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a + b)
print(a - b)
print(a * b)


# In[ ]:


#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a//b)
print(a/b)


# In[ ]:


#Python If-Else
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
if n % 2 == 1:
    print("Weird")
elif n % 2 == 0 and 2 <= n <= 5:
    print("Not Weird")
elif n % 2 == 0 and 6 <= n <= 20:
    print("Weird")
else:
    print("Not Weird") 


# In[ ]:


#Print Function
if __name__ == '__main__':
    n = int(input())
for i in range(1,n+1):
    print(i, end='')


# In[ ]:


#Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(0,n):
         print(i**2)


# In[ ]:


#Write a function
def is_leap(year):
    leap = False
    
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False
    
    return leap


# In[ ]:


#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    l = []
    for a in range(x + 1 ):
        for b in range(y + 1):
            for c in range(z + 1):
                if a + b + c != n:
                    l.append([a,b,c])
    print(l)


# In[ ]:


#Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    list1 = list(set(arr))
    list1.sort()
    print(list1[-2])


# In[ ]:


#Nested Lists
if __name__ == '__main__':
    records = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        student = [name,score]
        records.append(student)
    
    grades = [student[1] for student in records]
    grades_unique = list(set(grades))
    grades_unique.sort()
    second_lowest_grade = grades_unique[1]
    students = sorted(student[0] for student in records if second_lowest_grade == student[1])
    print('\n'.join(students))
        


# In[ ]:


#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    l = len(student_marks[query_name])
    s = sum(student_marks[query_name])
    a = s/l
    print("{:.2f}".format(a))


# In[ ]:


#Lists
if __name__ == '__main__':
    N = int(input())
    l = []
    for i in range(N):
        a = input().split()
        if a[0] == "insert":
            l.insert(int(a[1]),int(a[2]))
        elif a[0] == "remove":
            l.remove(int(a[1]))
        elif a[0] == "append":
            l.append(int(a[1]))
        elif a[0] == "sort":
            l.sort()
        elif a[0] == "pop":
            l.pop()
        elif a[0] == "reverse":
            l.reverse()
        else:
            print(l)


# In[ ]:


#Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    integer_list = tuple(integer_list)
    hashed = hash(integer_list)
    print(hashed)  


# In[ ]:


#Arrays
def arrays(arr):
    arr.reverse()
    return numpy.array(arr, float)


# In[ ]:


#Shape and Reshape
import numpy as np
n = input().split()
list = []
for i in n:
    list.append(int(i))
list = np.array(list)
list = list.reshape(3,3)
print(list)


# In[ ]:


#Transpose and Flatten
import numpy as np
n,m = list(map(int, input().split()))
list1 = []
for i in range(n):
    list1.append(input().split())
list2 = np.array(list1,int)
list1 = np.array(list1,int)
list2 = list2.transpose()
print(list2)
list1 = list1.flatten()
print(list1)


# In[ ]:


#Concatenate
import numpy as np

n, m, p = map(int, input().split())
nexp = []
mexp = []
for _ in range(n):
    nexp.append(np.array(list(map(int,input().split()))))

for _ in range(m):
    mexp.append(np.array(list(map(int,input().split()))))

print(np.concatenate((nexp,mexp), axis = 0))


# In[ ]:


#Zeros and Ones
import numpy as np
n = list(map(int,input().split()))
print(np.zeros(n,int), np.ones(n,int), sep="\n")


# In[ ]:


#Eye and Identity
import numpy as np
np.set_printoptions(legacy='1.13')
print(np.eye(*map(int,input().split())))


# In[ ]:


#Array Mathematics
import numpy as np

N,M = list(map(int,input().split()))
A,B = [[input().split() for _ in range(N)] for _ in range(2)]
arrA, arrB = np.array(A,int), np.array(B,int)
addAB = arrA + arrB
subAB = arrA - arrB
prodAB = arrA * arrB
divAB = arrA // arrB
modAB = arrA % arrB
expAB = arrA ** arrB
print(addAB,subAB,prodAB,divAB,modAB,expAB,sep='\n')


# In[ ]:


#Floor, Ceil and Rint
import numpy as np
np.set_printoptions(legacy='1.13')

array = np.array(input().split(),float)
function = [np.floor, np.ceil, np.rint]
for fn in function:
    print(fn(array))


# In[ ]:


#Sum and Prod
import numpy as np
n,m = map(int, input().split())
a = np.array([list(map(int,input().split())) for _ in range(n)], np.int64)
print(np.prod(np.sum(a, axis=0)))


# In[ ]:


#and Max
import numpy as np
n,m = list(map(int,input().split()))
list = []
for i in range(n):
    list.append(input().split())
list = np.array(list,int)
list = np.min(list, axis = 1)
list = np.max(list)
print(list)


# In[ ]:


#Mean, Var, and Std
import numpy as np
n,m = list(map(int,input().split()))
array = np.array([input().split() for _ in range (n)], int)
print(np.mean(array,axis = 1), np.var(array,axis = 0), np.around(np.std(array),11), sep = "\n")


# In[ ]:


#Dot and Cross
import numpy as np
n = int(input())
a = np.array([input().split() for _ in range(n)], int)
b = np.array([input().split() for _ in range(n)], int)
print(np.dot(a,b))


# In[ ]:


#Inner and Outer
import numpy as np
a = list(map(int,input().split()))
b = list(map(int,input().split()))
A = np.array(a)
B = np.array(b)
print(np.inner(A,B))
print(np.outer(A,B))


# In[ ]:


#Polynomials
import numpy as np
P = list(map(float,input().split()))
x = float(input())

print(float(np.polyval(P,x)))


# In[ ]:


#Linear Algebra
import numpy as np
n = int(input())
a = np.array([input().split() for _ in range(n)], float)
np.set_printoptions(legacy='1.13')
print(np.linalg.det(a))


# In[ ]:


#XML 1 - Find the Score
def get_attr_number(node):
    attr_number = len(node.attrib)
    for child in node.findall('.//'):
        attr_number += len(child.attrib)
    return attr_number


# In[ ]:


#XML2 - Find the Maximum Depth
maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1
    for child in elem:
        depth(child, level +1)


# In[ ]:


#Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        f('+91 {} {}'.format(n[-10:-5], n[-5:]) for n in l)
    return fun


# In[ ]:


#Decorators 2 - Name Directory

def person_lister(f):
    def inner(people):
        for ps in sorted(people, key = lambda x: int(x[2])):
            yield f(ps)
    return inner


# In[ ]:


#Map and Lambda Function

cube = lambda x: x**3# complete the lambda function 

def fibonacci(n):
    f0, f1 = 0, 1
    fib = []
    for _ in range(n):
        fib.append(f0)
        f0, f1 = f1, f0 + f1
    return fib
    
    # return a list of fibonacci numbers


# In[ ]:


#Validating Email Addresses With a Filter

import re
def fun(s):
    if re.search("^[\w-]+@[a-zA-Z0-9]+\.[a-zA-Z]{1,3}$",s):
        return True
    else:
        return False
    
    # return True if s is a valid email, else return False


# In[ ]:


#Reduce Function

def product(fracs):
    t = reduce(lambda x,y:x*y, fracs)# complete this line with a reduce statement
    return t.numerator, t.denominator


# In[ ]:


#Zipped!

# Enter your code here. Read input from STDIN. Print output to STDOUT
N,X = map(int,input().split())

a = [list(map(float,input().split())) for i in range(X)]
b = [print(round(sum(i)/len(i),1)) for i in list(zip(*a))]


# In[ ]:


#Input()

# Enter your code here. Read input from STDIN. Print output to STDOUT
x,k = map(int,input().split())
print(k == eval(input()))


# In[ ]:


#Python Evaluation

# Enter your code here. Read input from STDIN. Print output to STDOUT
eval(input().strip())


# In[ ]:


#Athlete Sort

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    for i in sorted(arr, key=lambda x:x[k]):
        print(*i)
    


# In[ ]:


#Any or All

# Enter your code here. Read input from STDIN. Print output to STDOUT
N = int(input())
integers = input().split()

if all(int(i) >= 0 for i in integers):
    if any(num == num[-1:] for num in integers):
        print("True")
    else:
        print("False")
else:
    print("False")


# In[ ]:


#ginortS

# Enter your code here. Read input from STDIN. Print output to STDOUT
s = list(input())
low = [i for i in s if i.isalpha()and i.islower()]
upper = [i for i in s if i.isalpha()and i.isupper()]
even = [i for i in s if i.isdigit() and int(i)%2 == 0]
odd = [i for i in s if i.isdigit() and not int(i)%2 == 0]
print("".join(sorted(low) + sorted(upper) + sorted(odd) + sorted(even)))


# In[ ]:


#Calendar Module

# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar as cal
m,d,y = map(int,input().split())
day = cal.weekday(year=y, month=m, day=d)
print(cal.day_name[day].upper())


# In[ ]:


#Time Delta

import math
import os
import random
import re
import sys
import datetime
import dateutil.parser

# Complete the time_delta function below.
def time_delta(t1, t2):
    t1 = dateutil.parser.parse(t1,fuzzy=True)
    t2 = dateutil.parser.parse(t2,fuzzy=True)
    return str(int(abs(t1-t2).total_seconds()))

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()


# In[ ]:


#sWAP cASE

def swap_case(s):
    s = s.swapcase()
    return s


# In[ ]:


#String Split and Join

def split_and_join(line):
    return "-".join(line.split())
    


# In[ ]:


#What's Your Name?

# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
    print(f"Hello {first} {last}! You just delved into python.")


# In[ ]:


#Mutations

def mutate_string(string, position, character):
    return string[:position] + character + string[position + 1:]


# In[ ]:


#Find a string

def count_substring(s, sb):
    return len([i for i in range(len(s)) if s[i:(i + len(sb))] ==sb])


# In[ ]:


#String Validators

if __name__ == '__main__':
    s = input()
    print(any([i.isalnum() for i in s]))
    print(any([i.isalpha() for i in s]))
    print(any([i.isdigit() for i in s]))
    print(any([i.islower() for i in s]))
    print(any([i.isupper() for i in s]))


# In[ ]:


#Text Alignment

#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))


# In[ ]:


#Text Wrap

def wrap(string, max_width):
    return '\n'.join(textwrap.wrap(string, max_width))


# In[ ]:


#Designer Door Mat

# Enter your code here. Read input from STDIN. Print output to STDOUT
s = input()
n = int(s[:(s.find(' '))])
m = n*3
num = 1
for row in range(n):
    if row < (int(n/2)):
        print(('.|.'* num).center(m, '-'))
        num += 2
    if row == (int(n/2)):
        print('WELCOME'.center(m, '-'))
    if row > (int(n/2)):
        num -= 2
        print(('.|.' * num).center(m, '-'))


# In[ ]:


String Formatting

def print_formatted(number):
    l = len(format(n, 'b'))
    for i in range(1, number+1):
        print(str(i).rjust(l), format(i, 'o').rjust(l), format(i,'X').upper().rjust(l), format(i, 'b').rjust(l), sep = ' ')


# In[ ]:


#Alphabet Rangoli

def print_rangoli(size):
    alphabet = ' abcdefghijklmnopqrstuvwxyz'

    for i in range(size,0,-1):
        c = alphabet[size:i:-1] + alphabet[i:size+1]
        c = '-'.join(c)
        print(c.center((size*4)-3,'-'))

    for i in range(0,size-1):
        c = alphabet[size:i+2:-1] + alphabet[i+2:size+1]
        c ='-'.join(c)
        print(c.center((size*4)-3,'-'))


# In[ ]:


#Capitalize!


# Complete the solve function below.
def solve(s):
    return ' '.join([i.capitalize() for i in s.split(' ')])


# In[ ]:


#The Minion Game

def minion_game(string):
    v = {'A', 'E', 'I', 'O', 'U'}
    s = zip(list(string), range(len(string), 0, -1))
    score = {'Kevin': 0, 'Stuart': 0}

    for element in s:
        if element[0] in v:
            score['Kevin'] += element[1]
        else:
            score['Stuart'] += element[1]

    if score['Kevin'] > score['Stuart']:
        print(f'Kevin {score["Kevin"]}')
    elif score['Kevin'] < score['Stuart']:
        print(f'Stuart {score["Stuart"]}')
    else:
        print('Draw')


# In[ ]:


#Merge the Tools!

def merge_the_tools(string, k):
    for i in range(0,len(string),k):
        sub_string=[]
        for m in range(i,i+k):
            if string[m] not in sub_string:
                sub_string.append(string[m])
        sub_seq=''.join(sub_string)
        print(sub_seq)
        sub_string.clear()


# In[ ]:


#Introduction to Sets

def average(array):
    return float(sum(set(array))/len(set(array)))


# In[ ]:


#Symmetric Difference

# Enter your code here. Read input from STDIN. Print output to STDOUT
M=int(input())
a=set([int(x) for x in input().split()])

N=int(input())
b=set([int(x) for x in input().split()])

for x in sorted(a^b):
    print(x)


# In[ ]:


#No Idea!

# Enter your code here. Read input from STDIN. Print output to STDOUT
h = 0
n, m = map(int, input().split())
arr = list(map(int, input().split()))
a = set(map(int, input().split()))
b = set(map(int, input().split()))
for i in arr:
    if i in a:
        h += 1
    if i in b:
        h -= 1
print(h)


# In[ ]:


#Set .add()

# Enter your code here. Read input from STDIN. Print output to STDOUT
N = int(input())
c = set()
for _ in range(N):
    c.add(input())

print(len(c))


# In[ ]:


#Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))

N = int(input())

for _ in range(N):
    cmd=input().split()
    if(len(cmd)==1):
        s.pop()
    else:
        cmd="s."+cmd[0]+"("+cmd[1]+")"
        eval(cmd)
        
print(sum(list(s)))


# In[ ]:


#Set .union() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT
input()
s1 = {s for s in input().split()}
input()
s2 = {s for s in input().split()}
print(len(s1.union(s2)))


# In[ ]:


#Set .intersection() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input()) 
n = set(map(int, input().split()))
b = int(input()) 
b = set(map(int, input().split()))
print (len(n.intersection(b)))


# In[ ]:


#Set .difference() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
a = set(map(int,input().split()))
m = int(input())
b = set(map(int,input().split()))
print(len(a.difference(b)))


# In[ ]:


#Set .symmetric_difference() Operation

# Enter your code here. Read input from STDIN. Print output to STDOUT
_, s1 = input(), set(input().split())
_, s2 = input(), set(input().split())
print(len(s1^s2))


# In[ ]:


#Set Mutations

# Enter your code here. Read input from STDIN. Print output to STDOUT
a = int(input())
set_A = set(map(int, input().split()))
N = int(input())
for _ in range(N):
    command, n = input().split()
    set_N = set(map(int, input().split()))
    getattr(set_A, command)(set_N)
print(sum(set_A))


# In[ ]:


#The Captain's Room

# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
n=int(input())
a=list(map(int,input().split()))
a=Counter(a)
for i in a:
    if a[i]!=n:
        print(i)


# In[ ]:


#Check Subset

# Enter your code here. Read input from STDIN. Print output to STDOUT
for _ in range(int(input())):
    len_a, a = int(input()), set(map(int,input().split()))
    len_b, b = int(input()), set(map(int, input().split()))
    print(a.issubset(b))


# In[ ]:


#Check Strict Superset

# Enter your code here. Read input from STDIN. Print output to STDOUT
n=set(map(int,input().split()))
m=int(input())
count=0
for x in range(m):
    a=set(map(int,input().split()))
    if len(n&a)==len(a):
        if len(n)-len(a)>0:
            count+=1
if count==m:
    print("True")
else:
    print("False")


# In[ ]:


#Company Logo

import math
import os
import random
import re
import sys
from collections import Counter



if __name__ == '__main__':
    s = input()
    c = Counter(sorted(s))
    for i,j in c.most_common(3):
        print(i,j)


# In[ ]:


#Word Order

# Enter your code here. Read input from STDIN. Print output to STDOUT
d={}
for _ in range(int(input())):
    s=input()
    d[s]=d.get(s,0)+1
print(len(d))
print(*d.values())


# In[ ]:


#Collections.deque()

# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
from operator import methodcaller
d = deque()
for _ in range(int(input())):
    f = methodcaller(*input().strip().split())
    f(d)
    
print(*d)


# In[ ]:


#Piling Up!

# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
for i in range(int(input())):
    k  = int(input())
    b = deque([int(a) for a in input().split()])
    res = []
    while b:
        if b[-1] >= b[0]:
            res.append(b.pop())
        else:
            res.append(b.popleft())
    if res == sorted(res,reverse=True):
        print('Yes')
    else:
        print('No')


# In[ ]:


#Validating Credit Card Numbers

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
for i in range(int(input())):
    card=input()
    pattern=r"^[4|5|6][0-9]{3}-?[0-9]{4}-?[0-9]{4}-?[0-9]{4}$"
    pattern_for_repeated_digits=r"([0-9])\1-?\1\1+"
    if re.search(pattern,card) and not re.search(pattern_for_repeated_digits, card):
        print("Valid")
    else:
        print("Invalid")


# In[ ]:


#Validating Postal Codes

regex_integer_in_range = r"_________"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"_________"	# Do not delete 'r'.
regex_integer_in_range = r"^[1-9][0-9]{5}$"
regex_alternating_repetitive_digit_pair = r"(?<=(\d)).(?=\1)"


# In[ ]:


#Validating Postal Codes

regex_integer_in_range = r"_________"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"_________"	# Do not delete 'r'.
regex_integer_in_range = r"^[1-9][0-9]{5}$"
regex_alternating_repetitive_digit_pair = r"(?<=(\d)).(?=\1)"


# In[ ]:


#Matrix Script

import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
a = list(zip(*matrix))
string = ""
for i in range(len(a)):
    string += "".join(a[i])
pattern = re.compile(r"(?<=\w)[!@#$%& ]{1,}(?=\s*\w)")
new_string = re.sub(pattern," ",string)
print(new_string)


# In[ ]:


#Validating and Parsing Email Addresses

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
print(*list(filter(lambda x : re.match(r"^<[a-zA-Z][\w\-\.\_]+@[a-zA-Z]+[.]+[a-zA-Z]{1,3}>",x.split()[1]),list(input() for i in range(int(input()))))),sep="\n")


# In[ ]:


#Hex Color Code

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

regex_pattern = r"#[0-9a-fA-F]{3,6}(?=[,|;|')'])"
s = ""
for _ in range(int(input())):
    s += input()
s=re.sub(r"[\n\t\s]*", "", s)
print(*re.findall(regex_pattern, s), sep='\n')


# In[ ]:


#HTML Parser - Part 1

# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser

class HParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start : "+tag)
        for attr in attrs:
            print("-> ", end = "")
            print(*attr, sep=" > ")
    def handle_endtag(self, tag):
        print("End   : "+tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty : "+tag)
        for attr in attrs:
            print("-> ", end = "")
            print(*attr, sep=" > ")

s = ""
for _ in range(int(input())):
    s += input()
parser = HParser()
parser.feed(s)


# In[ ]:


#HTML Parser - Part 2

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if data != '\n':
            if "\n" in data:
                print(">>> Multi-line Comment")
                print(data)
            else:
                print(">>> Single-line Comment")
                print(data)
    def handle_data(self, data):
        if not data == '\n':
            print(f">>> Data")
            print(data)
  
  

html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()


# In[ ]:


#Detect HTML Tags, Attributes and Attribute Values

# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
class CustomHtmlParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for ele in attrs:
            print("->", ele[0], ">", ele[1])


parser = CustomHtmlParser()
for _ in range(int(input())):
    parser.feed(input())


# In[ ]:


#Validating UID

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
for i in range(int(input())):
    s=input()
    d=''.join(sorted(s))
    p=re.search(r'[0-9]{3,}[A-Za-z]{2,}$',d)
    if p and len(set(s))==len(s)and len(s)==10:
         print('Valid')
    else:
        print('Invalid')
    


# In[ ]:


#DefaultDict Tutorial

# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict
n,m = map(int,input().split())
d = defaultdict(list)
for i in range(n):
    d[input()].append(i+1)
for i in range(m):
    r = input()
    res = d[r] if len(d[r])>0 else [-1]
    print(*res)


# In[ ]:


#collections.Counter()

# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
X = input()
x = input().split()
d = dict(Counter(x))
s = 0
for i in range(int(input())):
    key, value = input().split()
    if key in x and d[key] > 0:
        s += int(value)
        d[key] -= 1
print(s)


# In[ ]:


#Collections.namedtuple()

# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple

N = int(input())
order = input().split()
student = namedtuple('student', order)
s = 0
for i in range(N):
    inpu = input().split()
    stu = student(*inpu)
    s += int(stu.MARKS)
    
print(s/N)


# In[ ]:


#Collections.OrderedDict()

# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict

n, dct = int(input()), OrderedDict()

for _ in range(n):
    
    *name, price = tuple(map(str, input().split()))   
    name, price = " ".join(name), int(price)
    
    if name in dct:
        dct[name] += price
    else:
        dct[name] = price

for k, v in dct.items():
    print(k, v)


# In[ ]:


#Group(), Groups() & Groupdict()

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = input()
pattern = re.compile(r"([\dA-Za-z])(?=\1)")
s = pattern.search(n)
if s:
    print(s.group(1))
else:
    print(-1)


# In[ ]:


#Re.findall() & Re.finditer()

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
s = input()
pattern = re.compile(r"(?<![AEIOU])([AEIOU]{2,})(?![AEIOU]).", re.I)
fi = pattern.findall(s)
if fi:
    print(*fi, sep='\n')
else:
    print(-1)


# In[ ]:


#Re.start() & Re.end()

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
string= input()
pattern = re.compile(input())
match = pattern.search(string)
if not match: print("(-1, -1)")
while match:
    print(f"({match.start()}, {match.end()-1})")
    match = pattern.search(string,match.start() + 1)


# In[ ]:


#Regex Substitution

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
strings=list(input() for i in range(int(input())))
for i in strings:
    print(re.sub(r'(?<=\s)(\|\|)(?=\s)',r"or",re.sub(r'(?<=\s)(&&)(?=\s)',r"and",i)))


# In[ ]:


#Validating Roman Numerals

regex_pattern = r""	# Do not delete 'r'.
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"


# In[ ]:


#Validating phone numbers

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

for _ in range(int(input())):
    if re.match(r'^[789]\d{9}$', input()):
        print('YES')
    else:
        print('NO')


# In[ ]:


#Exceptions

# Enter your code here. Read input from STDIN. Print output to STDOUT
for _ in range(int(input())):
    try:
        a, b = map(int, input().split())
        print(a//b)

    except (ValueError, ZeroDivisionError) as error:
        print("Error Code:", error)


# In[ ]:


#Incorrect Regex

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

for _ in range(int(input())):
    regex_pattern = "r'" + input() + "'"
    try:
        re.compile(regex_pattern)
        print("True")
    except:
        print("False")


# In[ ]:


#Detect Floating Point Number

# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
n = int(input())
for i in range(n):
    if re.match('^[+-]{0,1}[\d]{0,}\.\d+$', input()):
        print(True)
    else:
        print(False)


# In[ ]:


#Re.split()

regex_pattern = r"[,.]"	# Do not delete 'r'.


# In[ ]:





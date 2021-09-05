import numpy as np
import math
import sympy
import os
import sys

debug = True

def getcofactor(m, i, j):
    return [row[: j] + row[j+1:] for row in (m[: i] + m[i+1:])]

def encrypt(key_matrix, plain_text):
    key_matrix = np.array(key_matrix)
    plain_text_matrix = np.array(plain_text)
    result = key_matrix.dot(plain_text_matrix)
    
    cipher_text = ""
    
    for j in range(len(result[0])):
        for i in range(len(result)):
            cipher_text += chr(result[i][j] % 26 + ord('A'))
    return cipher_text

# Function to find the gcd (using Euclidean algorithm)
def find_gcd(val1 , val2):
    while(val2):
        val1, val2 = val2, val1 % val2
    return val1

# Function to find the determinant of the matrix 
def find_determinant(mat):
 
    # if given matrix is of order 2*2 then simply return det
    # value by cross multiplying elements of matrix.
    if(len(mat) == 2):
        value = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]
        return value
 
    # initialize Sum to zero
    Sum = 0
 
    # loop to traverse each column of matrix a.
    for current_column in range(len(mat)):
 
        # calculating the sign corresponding to co-factor of that sub matrix.
        sign = (-1) ** (current_column)
 
        # calling the function recursily to get determinant value of sub matrix obtained.
        sub_det = find_determinant(getcofactor(mat, 0, current_column))
 
        # adding the calculated determinant value of particular column matrix to total Sum.
        Sum += (sign * mat[0][current_column] * sub_det)
 
    # returning the final Sum
    return Sum

'''
def find_determinant(m):
    matrix =  np.array(m)
    return np.linalg.det(matrix)
'''


# main function 
if __name__ == "__main__":
    
    # Take input form user using command line argument
    # python3 encrypt.py <key file> <plain-text file>
    
    if(len(sys.argv) != 3):
        print("# Wrong command line argument use the following format :")
        print("# python3 encrypt.py <key file> <plain-text file>")
        exit()
    
    key_loc = sys.argv[1]
    plain_loc = sys.argv[2]
    
    if(debug):
        print("# Location of key file :",key_loc)
        print("# Location of plain-text file :",plain_loc)
    
    # Checking if the path to the key file is correct or not !
    if (os.path.exists(key_loc)):
        key_file = open(key_loc,'r')
    else:
        print("# Key file does not exist !!!")
        exit()
    
    # Checking if the path to the plain-text file is correct or not !
    if(os.path.exists(plain_loc)):
        plain_file = open(plain_loc,'r')
    else:
        print("# Plain-text file does not exist !!!")
        exit()
    
    # Reading the key file
    if(debug):
        print("\n###### Reading the key file ######")
    
    key_len = int(key_file.readline())
    key = []
    for i in range(key_len):
        tmp = key_file.readline().upper().strip().split(" ")
        key.append(tmp)
    
    if(debug):
        print("# The size of the key is :",key_len)
        print("# Printing the key :")
        for i in range(key_len):
            print (key[i])
    
    # Reading the palin-text file
    if(debug):
        print("\n###### Reading the plain-text file ######")
    
    plain = ""
    tmp = plain_file.readline().upper()
    while(tmp):
        
        # Cleaning the text
        for i in range(len(tmp)):
            if(tmp[i]>='A' and tmp[i]<='Z'):
                plain = plain+tmp[i]
        
        tmp = plain_file.readline().upper()
    
    plain_len = len(plain)
    
    if(debug):
        print("# The size of the plain-text is :",plain_len)
        print("# Printing the palin-text :\n"+plain)
    
    # Processing the key
    # Changing the key to integer % 26 [taking english language A=0 , .... , Z=25]
    for i in range(key_len):
        for j in range (key_len):
            key[i][j] = ord(key[i][j])-ord('A')
    
    if(debug):
        print("\n# Key after changing to integer")
        print("# taking english language A=0 , .... , Z=25")
        for i in range(key_len):
            print(key[i])
    
    # Processing the plain-text
    # Changing the palin-text to integer % 26 [taking english language A=0 , .... , Z=25]
    
    # 1) checking the length of the plain text and adding padding character is required
    if(plain_len % key_len != 0):
        # Appending X at the end of the text
        plain = plain+'X'*(key_len-(plain_len % key_len))
    
    plain_len = len(plain)
    
    if(debug):
        print("\n# Length of plain-text after appending 'X' :",plain_len)
        print("# Plain-text after appending 'X' :")
        print(plain)
    
    # 2) Changing the plain-text to integer % 26 [taking english language A=0 , .... , Z=25]
    tmp = plain
    plain = []
    for i in range(plain_len):
        plain.append(ord(tmp[i])-ord('A'))
    
    if(debug):
        print("# Plain-text after changing to integer :")
        print(plain)
        
    # 3) Changing the plain-text to n*k array where n*n is the key and n*k = plain_len
    tmp = []
    for i in range(key_len):
        tmp.append([])
        
    i = 0
    while(i<plain_len):
        for j in range(key_len):
            tmp[j].append(plain[i])
            i+=1;
    plain = tmp
    
    if(debug):
        print("\n# Plain-text after changing grouping according to key :")
        for i in plain:
            print(i)
            
    # Checking the constraints on Key
    if(debug):
        print("# Checking the constraints on key :")
        
    # 1) Checking the determinant of the key
    det = find_determinant(key)
    
    if (debug):
        print("\n# Determinant of the key is :",det)
    
    if(det == 0):
        print("# Determinant of the key matrix is Zero so it is not possible to find the inverse of the key.")
        print("# Choose a key whose determinant is Zero")
        exit()
    
    # 2) Checking if determinant of the key and key_len are co-prime
    gcd = find_gcd(abs(det),key_len)
    
    if (debug):
        print("\n# GCD of the determinant of key and key_len is :",gcd)
    
    if(gcd != 1 ):
        print("# GCD of Determinant of the key matrix and key length is not 1. (Not co-prime)")
        print("# Choose a key whose determinant and key length are co-prime")
        exit ()
    
    cipher_text = encrypt(key,plain)
    
    print("# Cipher text is :")
    print(cipher_text)
    
    key_file.close()
    plain_file.close()
    
    cipher_file = open("cipher_text_encrypt.txt",'w')
    cipher_file.write(cipher_text)
    cipher_file.close()
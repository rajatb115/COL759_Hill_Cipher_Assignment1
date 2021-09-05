import numpy as np
import math
import sympy
import os
import sys

debug = True

def getcofactor(m, i, j):
    return [row[: j] + row[j+1:] for row in (m[: i] + m[i+1:])]

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

def adjoin(mat):
    
    ml = len(mat)
    
    ad = []
    
    for i in range(ml):
        tmp = []
        for j in range(ml):
            tmp.append([])
        ad.append(tmp)
    
    for i in range(ml):
        for j in range(ml):
            tmp = []
            for k in range(ml):
                
                if (k == i):
                    continue
                tmp2 = []
                
                for l in range(ml):
                    if (l == j):
                        continue
                    
                    tmp2.append(mat[k][l])
                tmp.append(tmp2)
            
            ad[j][i] = ((-1)**(i+j))*find_determinant(tmp)
    return ad


def decrypt(key_matrix, cipher_text):
    
    det = find_determinant(key_matrix)
    adj = adjoin(key_matrix)
    a3 = sympy.mod_inverse(det, 26)

    
    # Inverse of the key matrix
    key_matrix_inv = adj
    
    for i in range(len(key_matrix_inv)):
        for j in range(len(key_matrix_inv)):
            key_matrix_inv[i][j] = key_matrix_inv[i][j]*a3
    
    cipher_text_matrix = np.array(cipher_text)
    key_matrix_inv = np.array(key_matrix_inv)

    result = np.array(np.dot(key_matrix_inv, cipher_text_matrix))
	# result = result.tolist()
    
    plain_text = ""
    for j in range(len(result[0])):
        for i in range(len(result)):
            plain_text += chr(int(round(result[i][j], 0) % 26 + 65))
    return plain_text

# main function 
if __name__ == "__main__":
    
    # Take input form user using command line argument
    # python3 decrypt.py <key file> <cipher-text file>
    
    if(len(sys.argv) != 3):
        print("# Wrong command line argument use the following format :")
        print("# python3 decrypt.py <key file> <cipher-text file>")
        exit()
    
    key_loc = sys.argv[1]
    cipher_loc = sys.argv[2]
    
    if(debug):
        print("# Location of key file :",key_loc)
        print("# Location of cipher-text file :",cipher_loc)
    
    # Checking if the path to the key file is correct or not !
    if (os.path.exists(key_loc)):
        key_file = open(key_loc,'r')
    else:
        print("# Key file does not exist !!!")
        exit()
    
    # Checking if the path to the cipher-text file is correct or not !
    if(os.path.exists(cipher_loc)):
        cipher_file = open(cipher_loc,'r')
    else:
        print("# Cipher-text file does not exist !!!")
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
        print("\n###### Reading the cipher-text file ######")
    
    cipher = ""
    tmp = cipher_file.readline().upper()
    while(tmp):
        
        # Cleaning the text
        for i in range(len(tmp)):
            if(tmp[i]>='A' and tmp[i]<='Z'):
                cipher = cipher+tmp[i]
        
        tmp = cipher_file.readline().upper()
    
    cipher_len = len(cipher)
    
    if(debug):
        print("# The size of the cipher-text is :",cipher_len)
        print("# Printing the palin-text :\n"+cipher)
    
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
    
    # Processing the cipher-text
    # Changing the cipher-text to integer % 26 [taking english language A=0 , .... , Z=25]
    tmp = cipher
    cipher = []
    for i in range(cipher_len):
        cipher.append(ord(tmp[i])-ord('A'))
    
    if(debug):
        print("# Cipher-text after changing to integer :")
        print(cipher)
        
    # 2) Changing the cipher-text to n*k array where n*n is the key and n*k = cipher_len
    tmp = []
    for i in range(key_len):
        tmp.append([])
        
    i = 0
    while(i<cipher_len):
        for j in range(key_len):
            tmp[j].append(cipher[i])
            i+=1;
    cipher = tmp
    
    if(debug):
        print("\n# Cipher-text after changing grouping according to key :")
        for i in cipher:
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
    
    plain_text = decrypt(key,cipher)
    
    print("# Plain text is :")
    print(plain_text)
    
    key_file.close()
    cipher_file.close()
    
    plain_file = open("plain_text_decrypt.txt",'w')
    plain_file.write(plain_text)
    plain_file.close()
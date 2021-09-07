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


def find_ic(txt):
    
    # count the occurance of the characters
    cnt=[]
    for i in range(26):
        cnt.append(0)
    
    for i in range(len(txt)):
        cnt[ord(txt[i])-ord("A")]+=1
    
    txt_length = len(txt)
    
    # using formula IC = (sum_i (f_i*(f_i-1)))/(n*(n-1))
    
    IC = 0
    
    for i in range(26):
        IC = IC + cnt[i]*(cnt[i]-1)
    IC = IC/(txt_lenght * (txt_length - 1))
    
    return IC

def modInverse(a, m):
    for x in range(1, m):
        if (((a%m) * (x%m)) % m == 1):
            return x
    return -1



def find_key(key_size,cipher,plain):
    cipher_txt = []
    plain_txt = []
    
    for i in range(key_size**2):
        cipher_txt.append(ord(cipher[i])-ord("A"))
        plain_txt.append(ord(plain[i])-ord("A"))
        
    print(cipher_txt)
    print(plain_txt)
    
    # AX=B
    A = []
    
    for i in range(key_size):
        tmp = []
        for j in range(key_size):
            tmp.append(cipher_txt[i*key_size+j])
        A.append(tmp)
        
    print(A)
    # X = A-1 B mod 26
    
    # Find A-1 mod 26
    det = find_determinant(A)
    
    if(det == 0):
        return -1
    
    # Find A inverse mod 26
    adj = adjoin(A)
    while(det < 0):
        det+=26
        
    # Is it required ?
    gcd = find_gcd(det,26)
    if(gcd != 1):
        return -1
    
    ### what to do here ?
    a3 = sympy.mod_inverse(abs(det), 26)
    A_inv = adj
    
    for i in range(len(A_inv)):
        for j in range(len(A_inv)):
            A_inv[i][j] = A_inv[i][j]*a3
    
    A_inv_np = np.array(A_inv)
    
    X = []
    
    for i in range(key_size):
        B = []
        for j in range(key_size):
            B.append(plain_txt[i+key_size*j])
        B_np = np.array(B)
        
        result = np.array(np.dot(A_inv_np, B_np))
        X.append(result)
    
    for i in range(len(X)):
        for j in range(len(X[i])):
            X[i][j] = X[i][j]%26
        
    print(X)
    return X
# main function 
if __name__ == "__main__":
    
    # Take input form user using command line argument
    # python3 partb.py <cipher-text file> <plain-text file>
    
    if(len(sys.argv) != 3):
        print("# Wrong command line argument use the following format :")
        print("# python3 partb.py <cipher-text file> <plain-text file>")
        exit()
    
    plain_loc = sys.argv[2]
    cipher_loc = sys.argv[1]
    
    if(debug):
        print("# Location of key file :",plain_loc)
        print("# Location of cipher-text file :",cipher_loc)
    
    # Checking if the path to the plain-text file is correct or not !
    if (os.path.exists(plain_loc)):
        plain_file = open(plain_loc,'r')
    else:
        print("# Plain-text file does not exist !!!")
        exit()
    
    # Checking if the path to the cipher-text file is correct or not !
    if(os.path.exists(cipher_loc)):
        cipher_file = open(cipher_loc,'r')
    else:
        print("# Cipher-text file does not exist !!!")
        exit()
    
    # Reading the cipher-text file
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
    
    # Reading the plain-text file
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
        
    IC = []
    for i in range(2,5,1):
        key = find_key(i,cipher,plain)
        
        if(key == -1):
            IC.append(-1)
            continue
            
        print(key)
        
        continue
        
        decrypt_txt = decrypt_text(key, cipher)
        
        IC.append(find_ic(decrypt_txt))
        
    exit()
    
    # IC of english = 0.065
    key_size = 2
    closest_IC = abs(0.065-IC[0])
    for i in range(3):
        if(abs(0.065-IC[i])<closest_IC):
            closest_IC = abs(0.065-IC[i])
            key_size = i+2
    
    key = find_key(key_size,cipher,plain)
    print("Key size =",key_size)
    print("Key",key)
    
    
        
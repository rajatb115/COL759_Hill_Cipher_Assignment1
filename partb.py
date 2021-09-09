import numpy as np
import math
import sympy
import os
import sys

debug = False

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


def find_ic(k_txt):
    
    k_IC = []
    for k, txt in k_txt:
    
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
        IC = IC/(txt_length * (txt_length - 1))
        
        k_IC.append([k,IC])
    
    return k_IC

def modInverse(a, m):
    for x in range(1, m):
        if (((a%m) * (x%m)) % m == 1):
            return x
    return -1

def find_batch(plain_txt, cipher_txt, epoch, key_size):
    plain =[]
    cipher = []
    
    for i in range(epoch,epoch+key_size**2,1):
        plain.append(plain_txt[i])
        cipher.append(cipher_txt[i])
    
    return plain,cipher

# Function to find the  key
def find_key(key_size,cipher,plain):
    
    cipher_txt = []
    plain_txt = []
    
    # Convert the cipher text and plain text into integers
    for i in range(min(len(plain),len(cipher))):
        cipher_txt.append(ord(cipher[i])-ord("A"))
        plain_txt.append(ord(plain[i])-ord("A"))
    
    if(debug):
        print("cipher txt :",cipher_txt)
        print("plain txt :",plain_txt)
    
    possible_key = []
    
    epoch=0
    while(True):
        
        # reading the plain text batch and cipher text batch for current epoc
        batch_plain, batch_cipher = find_batch(plain_txt, cipher_txt, epoch, key_size)
        
        if (debug):
            print("\nbatch_plain:",batch_plain)
            print("batch_cipher:",batch_cipher)
        
        # AX=B
        A_plain = []
        B_cipher = []
        for i in range(key_size):
            tmp_cipher = []
            tmp_plain = []
            for j in range(key_size):
                tmp_cipher.append(batch_cipher[i+j*key_size])
                tmp_plain.append(batch_plain[i+j*key_size])
            A_plain.append(tmp_plain)
            B_cipher.append(tmp_cipher)
        
        if(debug):
            print("Printing A_plain :",A_plain)
            print("Printing B_cipher :",B_cipher)
        
        # X = B A-1 mod 26
        # Find A-1 mod 26
        # Testing determinent if det = 0 then the matrix is not invertible
        
        ###################################
        # To speed up the code
        #det = find_determinant(A_plain)
        
        A_tmp = np.array(A_plain)
        det = round(np.linalg.det(A_tmp))
        
        ###################################
        
        if(debug):
            print("det =",det)
        
        if(det == 0):
            epoch+=1
            if(len(plain) < epoch+key_size**2):
                break
            continue
        
        # Find A inverse mod 26
            
        gcd = find_gcd(abs(det),26)
        
        if(debug):
            print("gcd of det with 26=", gcd)
        
        if(gcd != 1):
            epoch+=1
            if(len(plain) < epoch+key_size**2):
                break
            continue
        
        a3 = sympy.mod_inverse(det, 26)
        
        if(debug):
            print("multiplicative inverse of det=",a3)
            
        
        #######################################
        # to speedup the code
        # adj = adjoin(A_plain)
        a1 = np.linalg.inv(A_tmp)
        adj = a1 * det
        
        #######################################
        
        A_inv = adj
        
        for i in range(len(A_inv)):
            for j in range(len(A_inv)):
                A_inv[i][j] = (round(A_inv[i][j])*a3 )%26
        
        
        
        A_inv_np = np.int_(np.array(A_inv))
        B_np = np.array(B_cipher)
        
        if(debug):
            print("A inv =",A_inv)
        
        X = np.array(np.dot(B_np, A_inv_np))
        
        key = []
        
        for i in range(len(X)):
            tmp = []
            for j in range(len(X[i])):
                X[i][j] = X[i][j]%26
                tmp.append(X[i][j])
            key.append(tmp)
        
        if key not in possible_key:
            possible_key.append(key)
        
        if(debug):
            print("key size :",key_size,"Key :",X)
        
        epoch+=1
        if(len(plain) < epoch+key_size**2):
            break
    
    return possible_key

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
    
    plain_text = ""
    for j in range(len(result[0])):
        for i in range(len(result)):
            plain_text += chr(int(round(result[i][j], 0) % 26 + 65))
    return plain_text


def decrypt_text(key, cipher):
    
    txt_key = []
    
    for k in key:
        
        
        cipher_txt = []
        for i in range(len(cipher)):
            cipher_txt.append(ord(cipher[i])-ord("A"))

        tmp1 = []

        for _ in range(len(k)):
            tmp1.append([])

        ii = 0
        while(ii<((len(cipher)//len(k))*len(k))):
            for j in range(len(k)):
                tmp1[j].append(cipher_txt[ii])
                ii+=1;

        cipher_mat = tmp1
        
        key_len = len(k)
        tmp = []
        
        det = find_determinant(k)
        if(det == 0):
            continue
        
        gcd = find_gcd(abs(det),26)
        if(gcd !=1):
            continue
        
        plain_text = decrypt(k,cipher_mat)
        tmp.append(k)
        tmp.append(plain_text)
        
        txt_key.append(tmp)
    return txt_key


def decrypt_plain(key, cipher):
    
    cipher_txt = []
    for i in range(len(cipher)):
        cipher_txt.append(ord(cipher[i])-ord("A"))

    tmp1 = []

    for _ in range(len(key)):
        tmp1.append([])
        
    ii = 0
    while(ii<((len(cipher)//len(key))*len(key))):
        for j in range(len(key)):
            tmp1[j].append(cipher_txt[ii])
            ii+=1;
    
    cipher_mat = tmp1
    
    key_len = len(key)
    tmp = []

    det = find_determinant(key)
    if(det == 0):
        return "wrong key : determinant is zero"

    gcd = find_gcd(abs(det),26)
    if(gcd !=1):
        return "wrong key : gcd of det and 26 is not 1 "
    
    return decrypt(key,cipher_mat)
    

# main function 
if __name__ == "__main__":
    
    # Take input form user using command line argument
    # python3 partb.py <cipher-text file> <plain-text file>
    
    if(len(sys.argv) != 3):
        print("# Wrong command line argument use the following format :")
        print("# python3 partb.py <cipher-text file> <plain-text file>")
        exit()
    
    cipher_loc = sys.argv[1]
    plain_loc = sys.argv[2]
    
    
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
    
    print("# The size of the cipher-text is :",cipher_len)
    print("# Printing the cipher-text :\n"+cipher)
    
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
    
    print("# The size of the plain-text is :",plain_len)
    print("# Printing the palin-text :\n"+plain)
        
    IC = []
    for i in range(2,11,1):
        
        # Find all the possible keys of each size
        key = find_key(i,cipher,plain)
        
        # Decrypt the cipher text using the expected keys
        decrypt_txt = decrypt_text(key, cipher)
        
        # check the IC of the decrypted text for the given keys
        IC1=find_ic(decrypt_txt)
        
        if(len(IC1)>0):
            for ic in IC1:
                IC.append(ic)
        
        if(debug):
            print(IC)
    
    # IC of english = 0.065
    key = IC[0][0]
    ICval = IC[0][1]
    closest_IC = abs(0.065-IC[0][1])
    for i in range(len(IC)):
        if(abs(0.065-IC[i][1])<closest_IC):
            closest_IC = abs(0.065-IC[i][1])
            key = IC[i][0]
            ICval = IC[i][1]
    
    # Print the results
    print("Key size =",len(key))
    print("Key :",key)
    print("IC_deviation from 0.065 :",closest_IC)
    print("IC value :",ICval)
    
    plain_decrypt = decrypt_plain(key,cipher)
    
    print("Decrypted Text :\n"+plain_decrypt)
    
    
    
        